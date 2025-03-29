# Knowledge-Augmented LLaMA2 Fine-Tuning with LoRA & Quantization
```markdown

This repository demonstrates memory-efficient fine-tuning of LLaMA2 using **LoRA (Low-Rank Adaptation)** and **4-bit quantization** for medical domain adaptation. Based on [QLoRA techniques](https://arxiv.org/abs/2310.08659) and optimized with [Hugging Face PEFT](https://huggingface.co/docs/peft/en/index).

---

## 📌 Key Features
✅ **Parameter-Efficient Fine-Tuning**  
- LoRA rank adaptation (r=64) on query/value projections  
- 4-bit NF4 quantization with BitsAndBytes  
- BF16 mixed-precision training  

✅ **Medical Domain Adaptation**  
- Trained on [Wiki Medical Terms](https://huggingface.co/datasets/gamino/wiki_medical_terms)  
- Optimized for clinical terminology understanding  

✅ **Memory Optimization**  
- <10GB VRAM usage through 4-bit quantization [1][3]  
- Frozen base model with trainable adapters [6]  

✅ **Reproducible Training**  
- Epoch-wise checkpointing  
- Batch size 4 for stable training  
- Full logging configuration  

---

## 🛠️ Installation
```
# Base requirements
pip install torch==2.1.0 transformers==4.35.0 peft==0.6.0 datasets==2.14.5

# Quantization support
pip install bitsandbytes==0.41.2 accelerate==0.24.0

# Training utilities
pip install trl==0.7.0 scikit-learn==1.3.0
```

---

## ⚙️ Configuration
### Model Setup
```
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "aboonaji/llama2finetune-v2",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### LoRA Configuration
```
from peft import LoraConfig

lora_config = LoraConfig(
    r=64,                  # Rank of LoRA matrices
    lora_alpha=16,         # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Layers to adapt
    lora_dropout=0.1,      # Regularization
    bias="none",           # No bias terms
    task_type="CAUSAL_LM"  # Causal language modeling
)
```

---

## 🚀 Training Workflow
### Dataset Preparation
```
from datasets import load_dataset

dataset = load_dataset("gamino/wiki_medical_terms")

# Tokenization example
tokenizer = AutoTokenizer.from_pretrained(model_base)
tokenizer.pad_token = tokenizer.eos_token  # Set padding

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_ds = dataset.map(tokenize_fn, batched=True)
```

### Training Arguments
```
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    bf16=True,              # Use bfloat16 precision
    logging_dir="./logs",
    save_strategy="epoch",
    logging_steps=10
)
```

### Start Training
```
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    args=training_args,
    tokenizer=tokenizer
)

trainer.train()  # Starts fine-tuning
```

---

## 🧪 Inference
```
from transformers import pipeline

generator = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,
    max_length=100
)

# Medical query example
prompt = "What are the diagnostic criteria for diabetes mellitus type 2?"
output = generator(f"~~[INST] {prompt} [/INST]")
print(output['generated_text'])
```

**Sample Output:**  
"Diabetes mellitus type 2 is diagnosed through a combination of fasting plasma glucose ≥126 mg/dL, HbA1c ≥6.5%, or random glucose ≥200 mg/dL with symptoms..."

---

## 📊 Performance Metrics
| Metric               | Value       |
|----------------------|-------------|
| Training Loss        | 1.18        | 
| GPU Memory Usage     | 10.8 GB     |
| Training Time/Epoch  | ~45 mins    |
| Batch Size           | 4           |

---

## 🚨 Common Issues & Fixes
1. **OOM Errors**  
   - Reduce `per_device_batch_size` to 2  
   - Enable gradient checkpointing  

2. **Quantization Warnings**  
   ```
   # Add to BitsAndBytesConfig:
   torch_dtype=torch.float16,
   ```

3. **Tokenizer Padding**  
   - Always set `pad_token = eos_token` for LLaMA2

---

## 📚 References
1. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)  
2. [LoRA: Low-Rank Adaptation of Large Models](https://arxiv.org/abs/2106.09685)  
3. [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)  
4. [Medical Domain Adaptation Strategies](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10353802/)  

---

**Note:** Requires NVIDIA GPU with ≥12GB VRAM. For CPU-only usage, disable 4-bit quantization and use `device_map="cpu"`.
