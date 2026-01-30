from llava.eval.model_vqa_loader import eval_model
from llava.eval.model_vqa_science import eval_model as eval_model_sqa
from llava.eval.eval_textvqa import eval_single
from llava.eval.eval_science_qa import eval_single_sqa
from llava.eval.eval_pope import eval_pope as eval_single_pope
from types import SimpleNamespace
from .eval_gqa import eval_single_gqa
import json
from collections import defaultdict
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}


class calculate_metrics:
    def divide_chunks(self, l, n=2):
        # looping till length l
        for i in range(0, len(l), n): 
            yield l[i:i + n]
        
        return 

    def parse_pred_ans(self, pred_ans):
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]

            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"

        return pred_label


    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            "yes": 1,
            "no": 0,
            "other": -1,
        }
        
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds) 

        clean_gts = []
        clean_preds = []
        other_num = 0 
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
        

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        return metric_dict


    def process_result(self, results_dir):
        to_return = 0
        for eval_type, task_name_list in eval_type_dict.items():
            print("===========", eval_type, "===========")
           
            scores = 0
            task_score_dict = dict()

            for task_name in task_name_list:

                task_txt = os.path.join(results_dir, task_name + ".txt")
                if os.path.exists(task_txt) == False:
                    continue
                has_task_involved = True
                lines = open(task_txt, 'r').readlines()
                chunk_lines = list(self.divide_chunks(lines)) # one image corresponds to two questions
                
                img_num = len(chunk_lines)
                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts = []
                preds = []

                for img_items in chunk_lines:
                    assert len(img_items) == 2
                    img_correct_num = 0

                    for img_item in img_items:
                        img_name, question, gt_ans, pred_ans = img_item.split("\t")

                        gt_ans = gt_ans.lower()
                        pred_ans = pred_ans.lower()

                        assert gt_ans in ["yes", "no"] # gt can only be yes or no.

                        pred_ans = self.parse_pred_ans(pred_ans)
                        assert pred_ans in ["yes", "no", "other"]

                        gts.append(gt_ans)
                        preds.append(pred_ans)
                        
                        if gt_ans == pred_ans:
                            img_correct_num += 1
                        
                        if pred_ans not in ["yes", "no"]:
                            task_other_ans_num += 1

                    if img_correct_num == 2:
                        acc_plus_correct_num += 1

                # cal TP precision acc, etc.
                metric_dict = self.compute_metric(gts, preds)
                acc_plus = acc_plus_correct_num / img_num
                metric_dict["acc_plus"] = acc_plus
                
                
                for k, v in metric_dict.items():
                    if k in ["acc", "acc_plus"]:
                        task_score += v*100
                
                task_score_dict[task_name] = task_score
                
                scores += task_score

            print("total score:", scores, "\n")
            for task_name, score in task_score_dict.items():
                print("\t", task_name, " score:", score)
            print("\n")
            to_return+=scores
        
        return to_return

project_dir = "/home/weiliu/student/xhm/LLaVA"

def get_gt(data_path):
    GT = {}
    for category in os.listdir(data_path):
        category_dir = os.path.join(data_path, category)
        if not os.path.isdir(category_dir):
            continue
        if os.path.exists(os.path.join(category_dir, 'images')):
            image_path = os.path.join(category_dir, 'images')
            qa_path = os.path.join(category_dir, 'questions_answers_YN')
        else:
            image_path = qa_path = category_dir
        assert os.path.isdir(image_path), image_path
        assert os.path.isdir(qa_path), qa_path
        for file in os.listdir(qa_path):
            if not file.endswith('.txt'):
                continue
            for line in open(os.path.join(qa_path, file)):
                question, answer = line.strip().split('\t')
                GT[(category, file, question)] = answer
    return GT

def convert_answer_to_mme(ckpt):
    GT = get_gt(
        data_path=f"{project_dir}/playground/data/eval/MME/MME_Benchmark_release_version"
    )

    experiment = ckpt

    result_dir = os.path.join(f"{project_dir}/playground/data/eval/MME/eval_tool", 'answers', experiment)
    os.makedirs(result_dir, exist_ok=True)

    answers = [json.loads(line) for line in open(os.path.join(f"{project_dir}/playground/data/eval/MME/answers", f'{experiment}.jsonl'))]

    results = defaultdict(list)
    for answer in answers:
        category = answer['question_id'].split('/')[0]
        file = answer['question_id'].split('/')[-1].split('.')[0] + '.txt'
        question = answer['prompt']
        results[category].append((file, answer['prompt'], answer['text']))

    for category, cate_tups in results.items():
        with open(os.path.join(result_dir, f'{category}.txt'), 'w') as fp:
            for file, prompt, answer in cate_tups:
                if 'Answer the question using a single word or phrase.' in prompt:
                    prompt = prompt.replace('Answer the question using a single word or phrase.', '').strip()
                if 'Please answer yes or no.' not in prompt:
                    prompt = prompt + ' Please answer yes or no.'
                    if (category, file, prompt) not in GT:
                        prompt = prompt.replace(' Please answer yes or no.', '  Please answer yes or no.')
                gt_ans = GT[category, file, prompt]
                tup = file, prompt, gt_ans, answer
                fp.write('\t'.join(tup) + '\n')

def eval_mme(ckpt, model):
    args = SimpleNamespace(model_path="liuhaotian/llava-v1.5-7b",
                           image_folder=f"{project_dir}/playground/data/eval/MME/MME_Benchmark_release_version",
                           question_file=f"{project_dir}/playground/data/eval/MME/llava_mme.jsonl",
                           answers_file=f"{project_dir}/playground/data/eval/MME/answers/{ckpt}.jsonl",
                           conv_mode="vicuna_v1",
                           temperature=0,
                           num_chunks=1,
                           chunk_idx=0,
                           top_p=None,
                           num_beams=1,
                           max_new_tokens=128,
                           flowcut=False)
    eval_model(args, model)
    convert_answer_to_mme(ckpt)
    cal = calculate_metrics()
    results_dir = f"{project_dir}/playground/data/eval/MME/eval_tool/answers/{ckpt}"
    return cal.process_result(results_dir)

def eval_pope(ckpt, model):
    args = SimpleNamespace(model_path="liuhaotian/llava-v1.5-7b",
                           image_folder=f"{project_dir}/playground/data/eval/pope/val2014",
                           question_file=f"{project_dir}/playground/data/eval/pope/llava_pope_test.jsonl",
                           answers_file=f"{project_dir}/playground/data/eval/pope/answers/{ckpt}.jsonl",
                           conv_mode="vicuna_v1",
                           temperature=0,
                           num_chunks=1,
                           chunk_idx=0,
                           top_p=None,
                           num_beams=1,
                           max_new_tokens=128,
                           flowcut=False,
                           eval_q_num=1000)
    eval_model(args, model)
    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.answers_file)]
    category = "adversarial"
    cur_answers = [x for x in answers if questions[x["question_id"]]["category"] == category]
    return eval_single_pope(cur_answers, f"{project_dir}/playground/data/eval/pope/coco/coco_pope_{category}.json", eval_q_num=args.eval_q_num)


def eval_scienceqa(ckpt, model):
    args = SimpleNamespace(model_path="liuhaotian/llava-v1.5-7b",
                           image_folder=f"{project_dir}/playground/data/eval/scienceqa/images/test",
                           question_file=f"{project_dir}/playground/data/eval/scienceqa/llava_test_CQM-A.json",
                           answers_file=f"{project_dir}/playground/data/eval/scienceqa/answers/{ckpt}.jsonl",
                           conv_mode="vicuna_v1",
                           temperature=0,
                           num_chunks=1,
                           chunk_idx=0,
                           single_pred_prompt=True,
                           flowcut=False,
                           eval_q_num=1000)
    eval_model_sqa(args, model)
    sqa_eval_args = SimpleNamespace(base_dir=f"{project_dir}/playground/data/eval/scienceqa",
                                    result_file=args.answers_file,
                                    split="test",
                                    options=["A", "B", "C", "D", "E"])
    return eval_single_sqa(sqa_eval_args)


def eval_textvqa(ckpt, model):
    args = SimpleNamespace(model_path="liuhaotian/llava-v1.5-7b",
                           image_folder=f"{project_dir}/playground/data/eval/textvqa/train_images",
                           question_file=f"{project_dir}/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl",
                           answers_file=f"{project_dir}/playground/data/eval/textvqa/answers/{ckpt}.jsonl",
                           conv_mode="vicuna_v1",
                           temperature=0,
                           num_chunks=1,
                           chunk_idx=0,
                           top_p=None,
                           num_beams=1,
                           max_new_tokens=128,
                           flowcut=False,
                           eval_q_num=1000)
    eval_model(args, model)
    return eval_single(f"{project_dir}/playground/data/eval/textvqa/TextVQA_0.5.1_val.json", args.answers_file, eval_q_num=args.eval_q_num)

def convert_gqa_for_eval(src, dst):
    all_answers = []
    for _, line in enumerate(open(src)):
        res = json.loads(line)
        question_id = res['question_id']
        text = res['text'].rstrip('.').lower()
        all_answers.append({"questionId": question_id, "prediction": text})

    with open(dst, 'w') as f:
        json.dump(all_answers, f)

def eval_gqa(ckpt, model):
    args = SimpleNamespace(model_path="liuhaotian/llava-v1.5-7b",
                           image_folder=f"{project_dir}/playground/data/eval/gqa/data/images",
                           question_file=f"{project_dir}/playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl",
                           answers_file=f"{project_dir}/playground/data/eval/gqa/answers/llava_gqa_testdev_balanced/{ckpt}.jsonl",
                           conv_mode="vicuna_v1",
                           temperature=0,
                           num_chunks=1,
                           chunk_idx=0,
                           top_p=None,
                           num_beams=1,
                           max_new_tokens=128,
                           flowcut=False,
                           eval_q_num=1000)
    eval_model(args, model)
    convert_gqa_for_eval(src=f"{project_dir}/playground/data/eval/gqa/answers/llava_gqa_testdev_balanced/{ckpt}.jsonl",
                         dst=f"{project_dir}/playground/data/eval/gqa/data/testdev_balanced_predictions.json")
    gqa_eval_args = SimpleNamespace(questions=f"{project_dir}/playground/data/eval/gqa/data/testdev_balanced_questions.json",
                                    predictions=f"{project_dir}/playground/data/eval/gqa/data/testdev_balanced_predictions.json",
                                    consistency=False,
                                    grounding=False)
    return eval_single_gqa(gqa_eval_args)