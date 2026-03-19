# based on tiger lab general verifier, but only verify the final answer
import json
from vllm import LLM, SamplingParams
import argparse
from modelscope import AutoTokenizer
import os

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, default='V2_test.json')
parser.add_argument('--save_name', type=str, default=None)
parser.add_argument('--aim_gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()
if args.save_name is None:
    args.save_name = os.path.splitext(os.path.basename(args.test_file))[0]
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.aim_gpu)

# Replace with your model path
model_path = 'general_verifier'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LLM(model=model_path, tensor_parallel_size=1,trust_remote_code=True, gpu_memory_utilization=0.9)

test_path = args.test_file
test_data = []
with open(test_path, 'r') as f:
    test_data = json.load(f)

# Prepare all prompts
all_prompts = []
for data_idx, data in enumerate(test_data):
    question = data['question']
    if 'target' in data:
        ground_truth = data['target']
    else:
        ground_truth = data['ground_truth']
    if 'response' in data:
        student_answer = data['response']
    elif 'answer' in data:
        student_answer = data['answer']
    else:
        student_answer = data['final_answer']

    if 'Final Answer' in student_answer:
        student_answer = student_answer.split('Final Answer')[-1].strip()
    elif 'Final Result' in student_answer:
        student_answer = student_answer.split('Final Result')[-1].strip()
    elif 'answer is' in student_answer.lower():
        student_answer = student_answer.lower().split('answer is')[-1].strip()
    elif '\\boxed{' in student_answer:
        student_answer = '\\boxed{' + student_answer.split('\\boxed{')[-1].strip()
    else:
        student_answer = student_answer[-1000:]

    # Create prompt
    prompt = (
        f"User: ### Question: {question}\n\n"
        f"### Ground Truth Answer: {ground_truth}\n\n"
        f"### Student Answer: {student_answer}\n\n"
        "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
        "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
        "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
    )
    all_prompts.append(prompt)

# Batch processing
test_res = []
all_wrong_data = []
batch_size = args.batch_size
sampling_params = SamplingParams(max_tokens=4096, n=1, logprobs=0, temperature=0.7)

for batch_start in range(0, len(all_prompts), batch_size):
    batch_end = min(batch_start + batch_size, len(all_prompts))
    batch_prompts = all_prompts[batch_start:batch_end]

    print(f"Processing batch {batch_start // batch_size + 1} / {(len(all_prompts) + batch_size - 1) // batch_size} ({batch_start}-{batch_end}/{len(all_prompts)})")

    try:
        outputs = model.generate(batch_prompts, sampling_params)

        for idx, output in enumerate(outputs):
            data_idx = batch_start + idx
            try:
                result_text = output.outputs[0].text.strip().lower()
                if 'yes' in result_text.split('final decision:')[-1].strip():
                    test_res.append(1)
                else:
                    test_res.append(0)
                    data = test_data[data_idx]
                    ground_truth = data.get('target', data.get('ground_truth'))
                    student_answer = data.get('response', data.get('answer', data.get('final_answer')))
                    all_wrong_data.append({'gt': ground_truth, 'answer': student_answer})
            except Exception as e:
                print(f"Error processing result {data_idx}: {e}")
                test_res.append(0)
    except Exception as e:
        print(f"Error processing batch {batch_start}-{batch_end}: {e}")
        for _ in range(len(batch_prompts)):
            test_res.append(0)

print("accuracy: ", sum(test_res) / len(test_res))

# Write correctness back into the original result file
for i, res in enumerate(test_res):
    test_data[i]['correct'] = bool(res)
with open(test_path, 'w') as f:
    json.dump(test_data, f, indent=4)

os.makedirs('res/eval/', exist_ok=True)
with open('res/eval/' + args.save_name + '.txt', 'w') as f:
    f.write(f"\n\ntest_data_path: {test_path}\n")
    f.write(f"accuracy: {sum(test_res) / len(test_res)}\n")
    f.write(f"test_res: {test_res}\n")