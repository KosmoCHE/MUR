import json
import os
import argparse


def getAnswer(response):
    if "\\boxed" in response[-50:]:
        pred = response.split("\\boxed")[-1]
    else:
        pred = response.split("answer is")[-1]
    for char in pred:
        if char.isupper() and char in ["A", "B", "C", "D", "E"]:
            return char

    return ""


parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, required=True)
parser.add_argument('--save_name', type=str, default=None)
args = parser.parse_args()
if args.save_name is None:
    args.save_name = os.path.splitext(os.path.basename(args.test_file))[0]

prediction = []
with open(args.test_file) as file:
    prediction = json.load(file)
print(len(prediction))
correct_num = 0
for i in range(len(prediction)):
    is_correct = False
    if "final_answer" in prediction[i]:
        response = prediction[i]['final_answer']
        pred = getAnswer(response)
        gt = prediction[i]['ground_truth']
        try:
            if gt == pred:
                correct_num += 1
                is_correct = True
        except:
            pass
    elif "all_answers" in prediction[i]:
        response = prediction[i]
        gt = prediction[i]['ground_truth']

        pred = getAnswer(response['all_answers'][0])
        if pred == gt:
            correct_num += 1
            is_correct = True
    else:
        response = prediction[i]
        gt = prediction[i]['ground_truth']
        if "all_answers" in response:
            all_ans = [0, 0, 0, 0]
            for each in response["all_answers"]:
                char = getAnswer(each)
                if char == "A":
                    all_ans[0] += 1
                elif char == "B":
                    all_ans[1] += 1
                elif char == "C":
                    all_ans[2] += 1
                elif char == "D":
                    all_ans[3] += 1
                if gt == "A":
                    num_gt = 0
                elif gt == "B":
                    num_gt = 1
                elif gt == "C":
                    num_gt = 2
                elif gt == "D":
                    num_gt = 3
            max_ans = max(all_ans)
            if all_ans.index(max_ans) == num_gt:
                correct_num += 1
                is_correct = True
    prediction[i]['correct'] = is_correct

# Write correctness back into the original result file
with open(args.test_file, 'w') as f:
    json.dump(prediction, f, indent=4)

accuracy = correct_num/len(prediction)
print(accuracy)
os.makedirs('res/eval/', exist_ok=True)
with open('res/eval/' + args.save_name + '.txt', 'w') as f:
    f.write(f"\n\ntest_data_path: {args.test_file}\n")
    f.write(f"accuracy: {accuracy}\n")
