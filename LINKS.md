# Lab 21 — Submission Links

**Học viên**: Hồ Đắc Toàn — 2A202600057

## GitHub Repository

https://github.com/dactoan10052004/2A202600057_HoDacToan_LAB21

## HuggingFace Hub — LoRA Adapter (r=16)

https://huggingface.co/dactoan123/lab21-llama3.2-3b-lora-r16

## Nội dung HF Hub

| File | Mô tả |
|------|-------|
| `adapter_model.safetensors` | LoRA adapter weights (r=16, alpha=32) |
| `adapter_config.json` | PEFT config — target modules, rank, alpha |
| `tokenizer.json` | Tokenizer (Llama-3.2-3B) |

## Cách load adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-3B-Instruct-bnb-4bit", load_in_4bit=True)
model = PeftModel.from_pretrained(base, "dactoan123/lab21-llama3.2-3b-lora-r16")
tokenizer = AutoTokenizer.from_pretrained("dactoan123/lab21-llama3.2-3b-lora-r16")
```
