# main.py

import os
import csv
import json
import glob
from typing import Dict, List
from scripts.prompt_constructor import build_psa_prompt
from scripts.attacker import Attacker
from scripts.evaluator import Evaluator

def load_jsonl_to_dict(jsonl_path: str) -> Dict[str, str]:
    section_dict = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            section = data.get("section", "")
            content = data.get("content", "")
            section_dict[section] = content
    return section_dict

def load_harmful_questions(csv_path: str) -> List[str]:
    questions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  
                questions.append(row[0])
    return questions

def main():
    attacker = Attacker()      # 读取 config 中的 model_type 等信息
    evaluator = Evaluator()    # 读取 config 中 judge 模型/或API的配置信息

    template_folder = os.path.join(os.getcwd(), "template")
    jsonl_files = glob.glob(os.path.join(template_folder, "*.jsonl"))

    harmful_questions_path = os.path.join(os.getcwd(), "harmful_questions", "harmful_questions.csv")
    harmful_questions = load_harmful_questions(harmful_questions_path)

    for jsonl_file in jsonl_files:
        base_name = os.path.basename(jsonl_file).replace(".jsonl", "")

        paper_summary = load_jsonl_to_dict(jsonl_file)

        csv_filename = f"{base_name}_template_judge.csv"
        output_csv_path = os.path.join(os.getcwd(), csv_filename)

        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=["question","answer","score","reason","template","category"])
            writer.writeheader()

            for question in harmful_questions:
                prompt_str = build_psa_prompt(paper_summary, question)

                answer_str = attacker.query_model(prompt_str)

                eval_result = evaluator.judge_answer(answer_str)

                row_data = {
                    "question": question,
                    "answer": answer_str,
                    "score": str(eval_result.get("score", "")),
                    "reason": eval_result.get("reason", ""),
                    "template": "PSA-Fixed-Template",  # 你可改为prompt_str或其他标识
                    "category": eval_result.get("category", "")
                }
                writer.writerow(row_data)

        print(f"【{base_name}】处理完成，结果已写入：{output_csv_path}")


if __name__ == "__main__":
    main()
