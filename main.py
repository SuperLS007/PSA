# main.py

import os
import csv
import json
import glob
import time
import argparse
import hashlib
from typing import Dict, List, Optional
from scripts.prompt_constructor import build_psa_prompt
from scripts.attacker import Attacker
from scripts.evaluator import Evaluator

def load_jsonl_to_dict(jsonl_path: str) -> Dict[str, str]:
    """加载JSONL文件到字典"""
    section_dict = {}
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                section = data.get("section", "")
                content = data.get("content", "")
                section_dict[section] = content
        return section_dict
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        return {}

def load_harmful_questions(csv_path: str) -> List[str]:
    """加载有害问题列表"""
    questions = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  
                    questions.append(row[0])
        return questions
    except Exception as e:
        print(f"Error loading harmful questions: {e}")
        return []

def generate_prompt_id(prompt_str: str) -> str:
    """生成Prompt的唯一标识符"""
    return hashlib.md5(prompt_str.encode('utf-8')).hexdigest()[:10]

def api_call_with_retry(func, *args, max_retries=3, retry_delay=5, **kwargs):
    """带重试功能的API调用"""
    retries = 0
    while retries < max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            retries += 1
            if retries < max_retries:
                print(f"API call error: {e}")
                print(f"Retrying in {retry_delay} seconds ({retries}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                print(f"Maximum retries reached, abandoning operation: {e}")
                return None

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PSA Paper Summary Attack Evaluation Tool")
    
    parser.add_argument("--template-dir", type=str, default="template",
                       help="Template directory path, defaults to 'template'")
    
    parser.add_argument("--questions-file", type=str, 
                       default="harmful_questions/harmful_behaviors.csv",
                       help="Harmful questions CSV file path")
    
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Output directory path, defaults to 'output'")
    
    parser.add_argument("--save-full-prompt", action="store_true",
                       help="Whether to save the full prompt content")
    
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit the number of questions processed, for testing")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    attacker = Attacker()      # 读取 config 中的 model_type 等信息
    evaluator = Evaluator()    # 读取 config 中 judge 模型/或API的配置信息

    # 获取绝对路径
    current_dir = os.getcwd()
    template_folder = os.path.join(current_dir, args.template_dir)
    jsonl_files = glob.glob(os.path.join(template_folder, "*.jsonl"))

    questions_path = os.path.join(current_dir, args.questions_file)
    harmful_questions = load_harmful_questions(questions_path)
    
    if args.limit and args.limit > 0:
        harmful_questions = harmful_questions[:args.limit]
        print(f"Limiting questions processed to: {args.limit}")

    if not jsonl_files:
        print(f"Warning: No .jsonl files found in {template_folder} directory")
        return

    if not harmful_questions:
        print(f"Warning: No harmful questions found in {questions_path}")
        return

    print(f"Found {len(jsonl_files)} summary files and {len(harmful_questions)} harmful questions")
    
    for jsonl_file in jsonl_files:
        base_name = os.path.basename(jsonl_file).replace(".jsonl", "")
        print(f"\nProcessing paper summary: {base_name}")

        paper_summary = load_jsonl_to_dict(jsonl_file)
        if not paper_summary:
            print(f"Skipping {base_name}: Unable to load summary content")
            continue

        csv_filename = f"{base_name}_template_judge.csv"
        output_csv_path = os.path.join(args.output_dir, csv_filename)

        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f_out:
            fieldnames = ["question", "answer", "score", "reason", "template", "category"]
            if args.save_full_prompt:
                fieldnames.append("full_prompt")
                
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            start_time = time.time()
            for idx, question in enumerate(harmful_questions, 1):
                question_start_time = time.time()
                print(f"Processing question {idx}/{len(harmful_questions)}: {question[:50]}...")
                
                # 构建PSA Prompt
                prompt_str = build_psa_prompt(paper_summary, question)
                prompt_id = generate_prompt_id(prompt_str)
                
                # 调用模型获取回答
                answer_str = api_call_with_retry(attacker.query_model, prompt_str)
                if answer_str is None:
                    answer_str = "Model call failed"
                
                # 评估回答
                eval_result = api_call_with_retry(evaluator.judge_answer, answer_str)
                if eval_result is None:
                    eval_result = {"score": 0, "reason": "Evaluation failed", "category": "Unknown"}

                # 准备数据行
                row_data = {
                    "question": question,
                    "answer": answer_str,
                    "score": str(eval_result.get("score", "")),
                    "reason": eval_result.get("reason", ""),
                    "template": f"PSA-{base_name}-{prompt_id}",  # 使用摘要名称和prompt哈希值
                    "category": eval_result.get("category", "")
                }
                
                if args.save_full_prompt:
                    row_data["full_prompt"] = prompt_str
                    
                writer.writerow(row_data)
                f_out.flush()  # 确保数据写入
                
                question_time = time.time() - question_start_time
                print(f"Question processing completed, time: {question_time:.2f} seconds, score: {row_data['score']}")

            total_time = time.time() - start_time
            print(f"[{base_name}] processing complete, total time: {total_time:.2f} seconds, results written to: {output_csv_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Program error: {e}")
