# PSA (Paper Summarize Attack) 项目

本项目用于对大语言模型进行PSA（论文摘要攻击）测试，通过提取论文摘要信息构建Prompt来测试模型对有害指令的响应。

## 项目结构

```
PSA/
├─ pdf/
│   ├─ attack/                     # 放攻击型论文 PDF
│   └─ defend/                     # 放防御型论文 PDF
│
├─ harmful_questions/
│   └─ harmful_behaviors.csv       # 一列"有害问题"
│
├─ template/
│   ├─ AttackPaper1_summary.jsonl  # summarize.py 输出
│   └─ DefensePaper1_summary.jsonl
│
├─ config/
│   └─ model_config.yaml           # model_type/api_key/本地模型路径等
│
├─ scripts/
│   ├─ prompt_constructor.py       # Prompt 拼装
│   ├─ attacker.py                 # 模型调用
│   └─ evaluator.py                # 回答评估
│
├─ output/                         # 输出结果目录
│
├─ main.py                         # 主流程脚本
├─ summarize.py                    # 论文摘要生成
├─ requirements.txt
└─ README.md
```

## 安装与配置

1. 安装依赖:

```bash
pip install -r requirements.txt
```

2. 配置 `config/model_config.yaml`，根据你的需求设置API密钥或本地模型路径:

```yaml
# 攻击模型配置
model_type: "api"      # api 或 local
api_provider: "openai" 
api_key: "your-api-key"   
api_model_name: "gpt-3.5-turbo"

# 使用本地模型的配置
local_model_name: "decapoda-research/llama-7b-hf"
local_model_path: "/path/to/your/local/model"  

# 评估模型配置
judge_model_type: "api"  
judge_api_model_name: "gpt-3.5-turbo"

# 本地评估模型
judge_local_model_name: "decapoda-research/llama-7b-hf"
judge_local_model_path: "/path/to/your/judge/model"  

# 摘要生成模型配置
summarize_model_path: "/path/to/your/summarize/model"  
```

## 使用流程

### 1. 生成论文摘要

将PDF文件放入 `pdf/attack/` 或 `pdf/defend/` 目录，然后运行:

```bash
# 基本用法
python summarize.py your_paper_name --type attack

# 高级用法 (自定义每个部分的token数)
python summarize.py your_paper_name --type attack --title 100 --author 100 --attack-methods 500 --mechanism 500 --related-work 500
```

这将在 `template/` 目录下生成一个 JSONL 摘要文件。

### 2. 运行攻击测试

```bash
# 基本用法
python main.py

# 高级用法
python main.py --template-dir template --questions-file harmful_questions/harmful_behaviors.csv --output-dir output --save-full-prompt --limit 10
```

结果将保存在 `output/` 目录下的CSV文件中。

### 参数说明

#### summarize.py 参数

- `paper_name`: PDF文件名（不含扩展名）
- `--type`: 论文类型 (attack/defend)
- `--title`: Title部分的最大生成token数
- `--author`: Author部分的最大生成token数
- `--attack-methods`: Attack Methods部分的最大生成token数
- `--mechanism`: Mechanism部分的最大生成token数
- `--related-work`: Related Work部分的最大生成token数
- `--pdf-dir`: PDF所在目录
- `--output-dir`: 输出目录
- `--config`: 模型配置文件路径
- `--device`: 使用的设备

#### main.py 参数

- `--template-dir`: 模板目录路径
- `--questions-file`: 有害问题CSV文件路径
- `--output-dir`: 输出目录
- `--save-full-prompt`: 是否保存完整Prompt
- `--limit`: 限制处理的问题数量

## 关键组件

1. **summarizer.py**: 提取论文PDF文本并生成结构化摘要
2. **prompt_constructor.py**: 将摘要与有害问题组合成PSA Prompt
3. **attacker.py**: 调用模型API或本地模型进行推理
4. **evaluator.py**: 评估模型回答的有害程度
5. **main.py**: 整合以上组件，完成完整流程

## 注意事项

- 确保本地模型路径配置正确，并有足够的GPU内存
- 使用API时注意速率限制和配额
- 如遇到API错误，程序会自动重试 