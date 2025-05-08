# scripts/attacker.py

import os
import yaml
import openai
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class Attacker:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(os.getcwd(), "config", "model_config.yaml")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model_type = self.config.get("model_type", "api")  

        if self.model_type == "api":
            self.api_provider = self.config.get("api_provider", "openai")
            self.api_key = self.config.get("api_key", None)
            if self.api_provider == "openai" and self.api_key:
                openai.api_key = self.api_key
            
        elif self.model_type == "local":
            self.local_model_name = self.config.get("local_model_name", "")
            
            model_path = self.config.get("local_model_path", self.local_model_name)
            
            print(f"Loading local model: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.gen_config = {
                "max_length": self.config.get("max_length", 1024),
                "temperature": self.config.get("temperature", 0.7),
                "top_p": self.config.get("top_p", 0.95),
                "do_sample": self.config.get("do_sample", True),
                "num_return_sequences": 1
            }

    def query_model(self, prompt: str) -> str:
        if self.model_type == "api":
            return self._query_api(prompt)
        elif self.model_type == "local":
            return self._query_local(prompt)
        else:
            return "Model type not supported."

    def _query_api(self, prompt: str) -> str:
        try:
            response = openai.ChatCompletion.create(
                model = self.config.get("api_model_name", "gpt-3.5-turbo"),
                messages = [{"role": "user", "content": prompt}],
                temperature = 0.7,
                max_tokens = self.config.get("max_tokens", 1000)
            )
            answer = response["choices"][0]["message"]["content"]
            return answer
        except Exception as e:
            print("Error calling API:", e)
            return f"Error calling API: {str(e)}"

    def _query_local(self, prompt: str) -> str:
        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            max_length = self.gen_config["max_length"]
            temperature = self.gen_config["temperature"]
            top_p = self.gen_config["top_p"]
            do_sample = self.gen_config["do_sample"]
            
            outputs = self.model.generate(
                **input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if answer.startswith(prompt):
                answer = answer[len(prompt):].strip()
                
            return answer
            
        except Exception as e:
            print(f"Local model generation error: {e}")
            return f"Local model generation error: {str(e)}"
