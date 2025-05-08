import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
import PyPDF2
import json
from tqdm import tqdm
import os
import sys
import argparse
import yaml

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + '\n'
                else:
                    print(f"Warning: No text extracted from page {page_num}.")
        return text
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""

def split_text_into_chunks(text, max_words=1000):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def check_if_relevant(model, tokenizer, chunk, section, device, max_new_tokens=256):
    prompt = (
        f"[INST] <<SYS>> You are an expert in academic writing. <</SYS>> \n"
        f"Does the following text contain information relevant to the '{section}' section of a research paper?\n"
        f"- If the section is 'Title', the title of the paper should contain in this chunk.\n"
        f"- If the section is 'Author',  author's name(s) should contain in this chunk.\n"
        f"- If the section is 'Mechanism analysis of successful jailbreak', analysis of why this attack method can success work(not attack method but why it works) should contain in this chunk.\n\n"
        f"{chunk}\n\n"
        "Please respond with 'Yes' or 'No'.[/INST]"
    )
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=tokenizer.model_max_length).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.95,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return "Yes" in response

def generate_content_for_section(model, tokenizer, chunk, section, device, max_new_tokens= 256):
    prompt = (
            f"[INST] <<SYS>> You are an expert in summarizing large language model jailbreak papers. <</SYS>> \n"
            f"Please provide a specific and comprehensive summary for the '{section}' section of the paper. The response should be tailored according to the content type of the section:\n"
            f"- If the section is 'Title', only provide the title of the paper.\n"
            f"- If the section is 'Author', only list the author's name(s).\n"
            f"- If the section is 'Mechanism analysis of successful jailbreak', you should analysis why this attack method success work.\n"
            f"- For other sections, provide a detailed summary relevant to the section's content.\n\n"
            f"Please begin with 'Sure, here is the summary for the {section}:' and ensure the response is appropriately formatted.\n\n"
            f"{chunk}\n\n"
            "Make sure the summary matches the specific section and its expected content.[/INST]"
    )

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=tokenizer.model_max_length).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()

def save_content_to_jsonl(content_dict, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for section, content in content_dict.items():
                entry = {
                    "section": section,
                    "content": content
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        print(f"Error saving JSONL file: {e}")
        return False

def load_model_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}

def parse_args():
    parser = argparse.ArgumentParser(description="Paper Summary Generator Tool")
    
    parser.add_argument("paper_name", type=str, help="PDF filename (without extension)")
    parser.add_argument("--title", type=int, default=100, help="Maximum tokens for Title section")
    parser.add_argument("--author", type=int, default=100, help="Maximum tokens for Author section")
    parser.add_argument("--attack-methods", type=int, default=500, help="Maximum tokens for Attack Methods section")
    parser.add_argument("--mechanism", type=int, default=500, help="Maximum tokens for Mechanism section")
    parser.add_argument("--related-work", type=int, default=500, help="Maximum tokens for Related Work section")
    
    parser.add_argument("--pdf-dir", type=str, default="pdf", help="PDF directory")
    parser.add_argument("--output-dir", type=str, default="template", help="Output directory")
    parser.add_argument("--type", type=str, default="", choices=["attack", "defend", ""], help="Paper type, default unspecified")
    
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Model config file path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use, such as 'cuda:0'")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    config = load_model_config(args.config)
    
    current_dir = os.getcwd()
    pdf_dir = os.path.join(current_dir, args.pdf_dir)
    
    if args.type:
        pdf_dir = os.path.join(pdf_dir, args.type)
    
    pdf_path = os.path.join(pdf_dir, f"{args.paper_name}.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file does not exist - {pdf_path}")
        return
    
    output_dir = os.path.join(current_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    output_jsonl_path = os.path.join(
        output_dir, 
        f"{args.paper_name}_{args.title}_{args.author}_{args.attack_methods}_{args.mechanism}_{args.related_work}.jsonl"
    )
    
    model_path = config.get("summarize_model_path", None)
    if not model_path:
        model_path = config.get("local_model_path", None)
    
    if not model_path:
        print("Error: No valid model path found in config file")
        return
    
    print(f"Using model path: {model_path}")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading tokenizer and model...")
    
    try:
        if "llama" in model_path.lower():
            tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                device_map=args.device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=args.device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return

    try:
        print(f"Extracting text from PDF: {pdf_path}")
        paper_text = extract_text_from_pdf(pdf_path)
        
        if not paper_text:
            print("Error: Cannot extract text from PDF or extracted text is empty")
            return
            
        print(f"Extracted {len(paper_text.split())} words from PDF.")

        print("Splitting text into chunks...")
        paper_chunks = split_text_into_chunks(paper_text)
        print(f"Text has been split into {len(paper_chunks)} chunks.")

        content_dict = {
            "Title": "",
            "Author": "",
            "Summary of Attack Methods": "",
            "Mechanism analysis of successful jailbreak": "",
            "Related Work": ""
        }

        sections_completed = {
            "Title": False,
            "Author": False,
            "Summary of Attack Methods": False,
            "Mechanism analysis of successful jailbreak": False,
            "Related Work": False
        }

        for chunk_idx, chunk in enumerate(paper_chunks):
            print(f"Processing chunk {chunk_idx+1}/{len(paper_chunks)}...")
            for section in content_dict.keys():
                if sections_completed[section]:
                    continue

                if section == "Title":
                    max_new_tokens = args.title
                elif section == "Author":
                    max_new_tokens = args.author
                elif section == "Summary of Attack Methods":
                    max_new_tokens = args.attack_methods
                elif section == "Mechanism analysis of successful jailbreak":
                    max_new_tokens = args.mechanism
                elif section == "Related Work":
                    max_new_tokens = args.related_work

                print(f"  Checking if chunk is relevant to '{section}'...")
                is_relevant = check_if_relevant(model, tokenizer, chunk, section, device)

                if is_relevant:
                    print(f"  Generating content for '{section}', max tokens={max_new_tokens}...")
                    section_content = generate_content_for_section(model, tokenizer, chunk, section, device, max_new_tokens)
                    content_dict[section] = section_content 
                    sections_completed[section] = True
                    print(f"  '{section}' section completed.")

        incomplete_sections = [section for section, completed in sections_completed.items() if not completed]
        if incomplete_sections:
            print(f"Warning: The following sections could not be generated: {', '.join(incomplete_sections)}")
        
        if save_content_to_jsonl(content_dict, output_jsonl_path):
            print(f"Paper content saved to {output_jsonl_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
