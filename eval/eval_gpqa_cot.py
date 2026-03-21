import json
import os
import argparse
from collections import Counter, defaultdict


def getAnswer(response):
    if "\\boxed" in response[-50:]:
        pred = response.split("\\boxed")[-1]
    else:
        pred = response.split("answer is")[-1]
    for char in pred:
        if char.isupper() and char in ["A", "B", "C", "D", "E"]:
            return char

    return ""


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
        with open(test_file) as f:
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


parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, required=True)
parser.add_argument('--save_name', type=str, default=None)
args = parser.parse_args()
if args.save_name is None:
    args.save_name = os.path.splitext(os.path.basename(args.test_file))[0]

prediction = load_data(args.test_file)
print(len(prediction))

# Group by question_idx for multi-rollout support
groups = defaultdict(list)
for i, item in enumerate(prediction):
    key = item.get('question_idx', i)
    groups[key].append(i)

num_questions = len(groups)
correct_num = 0

for q_idx, indices in groups.items():
    rollouts = [prediction[i] for i in indices]
    gt = rollouts[0]['ground_truth']

    if len(rollouts) == 1:
        # Single rollout: direct evaluation
        item = rollouts[0]
        is_correct = False
        if "final_answer" in item:
            pred = getAnswer(item['final_answer'])
            if gt == pred:
                correct_num += 1
                is_correct = True
        elif "all_answers" in item:
            pred = getAnswer(item['all_answers'][0])
            if pred == gt:
                correct_num += 1
                is_correct = True
        for i in indices:
            prediction[i]['correct'] = is_correct
    else:
        # Multi-rollout: majority voting
        answers = [getAnswer(r['final_answer']) for r in rollouts]
        answers_valid = [a for a in answers if a]
        is_correct = False
        if answers_valid:
            most_common = Counter(answers_valid).most_common(1)[0][0]
            if most_common == gt:
                correct_num += 1
                is_correct = True
        # Mark each rollout with per-rollout and majority-vote correctness
        for i, ans in zip(indices, answers):
            prediction[i]['correct'] = (ans == gt)
            prediction[i]['majority_vote_correct'] = is_correct

# Write correctness back into the original result file
save_data(args.test_file, prediction)

accuracy = correct_num / num_questions
print(f"accuracy: {accuracy} ({correct_num}/{num_questions})")
os.makedirs('res/eval/', exist_ok=True)
with open('res/eval/' + args.save_name + '.txt', 'w') as f:
    f.write(f"\n\ntest_data_path: {args.test_file}\n")
    f.write(f"accuracy: {accuracy}\n")
    f.write(f"num_questions: {num_questions}\n")
    f.write(f"num_rollouts_per_question: {len(prediction) // num_questions}\n")
