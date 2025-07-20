import torch
from torch.utils.data import Dataset
import pandas as pd
import copy
from tqdm import tqdm
from adapter import LLamaAdapter
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig, 
    HfArgumentParser, 
    BitsAndBytesConfig,
)

PROMPT_DICT = {
    "prompt_input": (
        "Given a buggy code and its patch, along with intermediate layer embeddings from a repair model, "
        "explain the bug and how the patch fixes it.\n\n"
        "### Buggy Code:\n{buggy_code}\n### Patch:\n{patch}\n\n### Explanation:"
    )
}
device = "cuda" if torch.cuda.is_available() else "cpu"

class FinetuneDataset(Dataset):
    def __init__(self, model: LLamaAdapter, dataframe_path: str, phase='train'):
        # --- Load data ---
        self.data = pd.read_csv(dataframe_path, nrows=500)
        required = ['buggy_code', 'fixed_code', 'gpt_explanation']
        if not all(c in self.data.columns for c in required):
            raise ValueError(f"CSV must contain columns: {required}")
        if self.data[required].isnull().any().any():
            raise ValueError("Found nulls in CSV.")

        # --- Tokenizers & config ---
        self.llama_tok = model.llama_tokenizer
        self.repair_tok = model.repairllama_tokenizer
        self.llama_max = model.llama_max_seq_len
        self.repair_max = model.repairllama_max_seq_len

        self.model = None
        if phase == "inference":
            self.model = model.repairllama

        # Ensure llama tokenizer has a pad token
        if self.llama_tok.pad_token_id is None:
            self.llama_tok.add_special_tokens({'pad_token': self.llama_tok.eos_token})
        self.llama_pad = self.llama_tok.pad_token_id

        self.phase = phase

    def __len__(self):
        return len(self.data)   

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        buggy = str(row['buggy_code'])
        fixed = str(row['fixed_code'])

        # --- RepairLlama side (always the same) ---
        repair_text = buggy + "\n// Fixed Code:\n" + fixed + self.repair_tok.eos_token
        repair_enc = self.repair_tok(
            repair_text,
            padding="max_length",
            max_length=self.repair_max,
            truncation=True,
            return_tensors="pt",
        )
        repair_input_ids = repair_enc.input_ids.squeeze(0)
        repairllama_mask = repair_enc.attention_mask.squeeze(0)

        # --- LLaMA side differs by phase ---
        # build the prompt (without explanation for inference)
        
        if self.phase == 'inference':
            prompt_text = PROMPT_DICT["prompt_input"].format(buggy_code=buggy, patch=fixed)
            # only encode prompt → we’ll generate from this
            explanation = str(row['gpt_explanation']) if 'gpt_explanation' in row and row['gpt_explanation'] is not None else ""
            llama_enc = self.llama_tok(
                prompt_text + self.llama_tok.eos_token,
                padding="max_length",
                max_length=self.llama_max,
                truncation=True,
                return_tensors="pt",
            )
            llama_input_ids = llama_enc.input_ids.squeeze(0)
            llama_mask      = llama_enc.attention_mask.squeeze(0)
            # no labels during inference
            return repair_input_ids, repairllama_mask, llama_input_ids, llama_mask, explanation

        else:
            inputs = self.repair_tok(buggy, return_tensors="pt")
            inputs_len = inputs["input_ids"].shape[1]
            inputs_ids = inputs["input_ids"].to(device)
            generation_config = GenerationConfig(
                num_beams=10,
                early_stopping=True,
            )

            outputs = self.model.generate(
                input_ids=inputs_ids,
                max_new_tokens=256,
                num_return_sequences=10,
                pad_token_id=self.repair_tok.pad_token_id,
                eos_token_id=self.repair_tok.eos_token_id,
                generation_config=generation_config,
            )

            output_ids = outputs[:, inputs_len:]
            output_patch = self.repair_tok.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            generated_fix = output_patch[0]
            prompt_text = PROMPT_DICT["prompt_input"].format(buggy_code=buggy, patch=generated_fix)
            # train/validation: include the explanation and build labels
            explanation = str(row['gpt_explanation'])
            full_text   = prompt_text + explanation + self.llama_tok.eos_token
            llama_enc = self.llama_tok(
                full_text,
                padding="max_length",
                max_length=self.llama_max,
                truncation=True,
                return_tensors="pt",
            )
            llama_input_ids = llama_enc.input_ids.squeeze(0)
            llama_mask      = llama_enc.attention_mask.squeeze(0)

            # mask out the prompt in the labels, so only the explanation gets learned
            expl_enc = self.llama_tok(
                explanation + self.llama_tok.eos_token,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            expl_len = expl_enc.input_ids.size(1)

            llama_labels = llama_input_ids.clone()
            llama_labels[:-expl_len] = -100

            return (
                repair_input_ids,
                repairllama_mask,
                llama_input_ids,
                llama_labels,
                llama_mask
            )
        