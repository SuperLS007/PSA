# scripts/attacker.py

import os
import yaml
import openai
from typing import Optional

# 如果你要用transformers:
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class Attacker:
    def __init__(self, config_path: Optional[str] = None):
        """
        读取model_config.yaml，初始化本地或API模型
        """
        if config_path is None:
            config_path = os.path.join(os.getcwd(), "config", "model_config.yaml")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model_type = self.config.get("model_type", "api")  # "local" or "api"

        # 如果是API(比如OpenAI)
        if self.model_type == "api":
            self.api_provider = self.config.get("api_provider", "openai")
            self.api_key = self.config.get("api_key", None)
            if self.api_provider == "openai" and self.api_key:
                openai.api_key = self.api_key
            # 其他API（比如Anthropic Claude）可在此扩展
            
        # 如果是本地模型
        elif self.model_type == "local":
            self.local_model_name = self.config.get("local_model_name", "")
            # 这里演示一个非常简化的加载，如果要真正加载大的LLM，需要更多逻辑
            # self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_name)
            # self.model = AutoModelForCausalLM.from_pretrained(self.local_model_name)
            # self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0)

    def query_model(self, prompt: str) -> str:
        """
        输入Prompt，返回模型回答
        """
        if self.model_type == "api":
            return self._query_api(prompt)
        elif self.model_type == "local":
            return self._query_local(prompt)
        else:
            return "Model type not supported."

    def _query_api(self, prompt: str) -> str:
        """
        简化示例：以OpenAI API为例
        """
        try:
            response = openai.ChatCompletion.create(
                model = self.config.get("api_model_name", "gpt-3.5-turbo"),
                messages = [{"role": "user", "content": prompt}],
                temperature = 0.7
            )
            answer = response["choices"][0]["message"]["content"]
            return answer
        except Exception as e:
            print("Error calling API:", e)
            return "Error calling API."

    def _query_local(self, prompt: str) -> str:
        """
        如果是本地模型(示例中只是简单return, 
        你可以用HuggingFace pipeline之类来完成真正推理)
        """
        # 伪代码:
        # result = self.pipe(prompt, max_length=1024, do_sample=True)
        # answer = result[0]["generated_text"]
        # 这里仅演示，所以返回固定字符串:
        return "LOCAL_MODEL_ANSWER (示例占位)"
