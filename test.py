from .adapter import LLamaAdapter
import argparse
import torch
from trnaformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/Llama3/Llama3.1-8B-Instruct")
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    repairllama_model_dir = "/content/drive/MyDrive/workspace/CodeLlama-7b-hf"
    repairllama_tokenizer = AutoTokenizer.from_pretrained(repairllama_model_dir, trust_remote_code=True, padding_size='left')

if __name__ == "__main__":
    main()