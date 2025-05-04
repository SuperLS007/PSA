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
    """
    读取 summarizer.py 生成的 JSONL 文件，
    将 {section, content} 结构存到一个字典里： {section_name: content_string}
    """
    section_dict = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            section = data.get("section", "")
            content = data.get("content", "")
            section_dict[section] = content
    return section_dict

def load_harmful_questions(csv_path: str) -> List[str]:
    """
    从 harmful_questions.csv 中读取所有有害问题，存成列表
    假定 csv 中只有一列：question
    """
    questions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # 如果 csv 有表头，则先 next(reader) 跳过表头
        # next(reader)
        for row in reader:
            if row:  # 取第一列作为问题
                questions.append(row[0])
    return questions

def main():
    # 第一步：初始化 Attacker 与 Evaluator
    attacker = Attacker()      # 读取 config 中的 model_type 等信息
    evaluator = Evaluator()    # 读取 config 中 judge 模型/或API的配置信息

    # 第二步：遍历 template 文件夹下所有jsonl（每个jsonl对应一篇论文）
    template_folder = os.path.join(os.getcwd(), "template")
    jsonl_files = glob.glob(os.path.join(template_folder, "*.jsonl"))

    # 第三步：读取有害问题
    harmful_questions_path = os.path.join(os.getcwd(), "harmful_questions", "harmful_questions.csv")
    harmful_questions = load_harmful_questions(harmful_questions_path)

    for jsonl_file in jsonl_files:
        # 取论文名（去掉后缀.jsonl）
        base_name = os.path.basename(jsonl_file).replace(".jsonl", "")

        # 加载该论文的摘要信息
        paper_summary = load_jsonl_to_dict(jsonl_file)

        # 打开一个 csv 文件来写结果
        csv_filename = f"{base_name}_template_judge.csv"
        output_csv_path = os.path.join(os.getcwd(), csv_filename)

        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=["question","answer","score","reason","template","category"])
            writer.writeheader()

            # 针对 harmful_questions 里的所有问题一一拼接prompt并攻击
            for question in harmful_questions:
                # 第1步：构造Prompt
                prompt_str = build_psa_prompt(paper_summary, question)

                # 第2步：向目标模型发送
                answer_str = attacker.query_model(prompt_str)

                # 第3步：评估回答
                eval_result = evaluator.judge_answer(answer_str)

                # 第4步：写入csv
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
