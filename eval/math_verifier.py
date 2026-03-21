# based on tiger lab general verifier, but only verify the final answer
import json
from vllm import LLM, SamplingParams
import argparse
from modelscope import AutoTokenizer
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, default='V2_test.json')
parser.add_argument('--save_name', type=str, default=None)
parser.add_argument('--aim_gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--verifier', type=str, required=True, help='Path to the verifier model')
args = parser.parse_args()
if args.save_name is None:
    args.save_name = os.path.splitext(os.path.basename(args.test_file))[0]
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.aim_gpu)

model_path = args.verifier

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LLM(model=model_path, tensor_parallel_size=1,trust_remote_code=True, gpu_memory_utilization=0.9)


def load_data(test_file):
    """Load data from JSONL or JSON format."""
    if test_file.endswith('.jsonl'):
        data = []
        with open(test_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    else:
        with open(test_file, 'r') as f:
            return json.load(f)


def save_data(test_file, data):
    """Save data back in the same format (JSONL or JSON)."""
    if test_file.endswith('.jsonl'):
        with open(test_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        with open(test_file, 'w') as f:
            json.dump(data, f, indent=4)


test_path = args.test_file
test_data = load_data(test_path)

# Prepare all prompts (one per rollout line)
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

# Group by question_idx for multi-rollout majority voting
groups = defaultdict(list)
for i, item in enumerate(test_data):
    key = item.get('question_idx', i)
    groups[key].append(i)

num_questions = len(groups)
correct_num = 0

for q_idx, indices in groups.items():
    if len(indices) == 1:
        # Single rollout
        i = indices[0]
        test_data[i]['correct'] = bool(test_res[i])
        correct_num += test_res[i]
    else:
        # Multi-rollout: majority voting on verification results
        votes = [test_res[i] for i in indices]
        majority_correct = sum(votes) > len(votes) / 2
        if majority_correct:
            correct_num += 1
        for i in indices:
            test_data[i]['correct'] = bool(test_res[i])
            test_data[i]['majority_vote_correct'] = majority_correct

accuracy = correct_num / num_questions
print(f"accuracy: {accuracy} ({correct_num}/{num_questions})")

# Write correctness back into the original result file
save_data(test_path, test_data)

os.makedirs('res/eval/', exist_ok=True)
with open('res/eval/' + args.save_name + '.txt', 'w') as f:
    f.write(f"\n\ntest_data_path: {test_path}\n")
    f.write(f"accuracy: {accuracy}\n")
    f.write(f"num_questions: {num_questions}\n")
    f.write(f"num_rollouts_per_question: {len(test_data) // num_questions}\n")
    f.write(f"test_res: {test_res}\n")
