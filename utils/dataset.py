# import torch
# from torch.utils.data import Dataset
# # import llama.utils
# # from codellama.tokenizer import Tokenizer
# import copy
# import pandas as pd
# from dataclasses import dataclass
# import sys
# from ..adapter import LLamaAdapter

# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\nThere is a buggy code provided and fixed code embeddings come through intermediate layers. write an explanation explaining bug and fix\n\n### Input:\n{buggy_code}\n\n### Response:"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:"
#     ),
# }

# class FinetuneDataset(Dataset):
#     def __init__(self, llama_tokenizer, repairllama_tokenizer, dataframe_path:str, phase='train'):# model:LLamaAdapter
#         print(f"read dataset  from {dataframe_path}")
#         self.data = pd.read_csv(dataframe_path)  # Load DataFrame from CSV file assumed have buggy_code, fixed_code and explanation columns
#         self.llama_tokenizer = llama_tokenizer
#         self.repairllama_tokenizer = repairllama_tokenizer
#         self.repairllama_max_input_len = 1024 # model.llama_max_seq_len
#         self.llama_max_input_len = 1024 # model.llama_max_seq_len
#         self.llama_pad_id = llama_tokenizer.pad_token_id#model.llama_tokenizer.pad_id
#         self.phase=phase
#         # self.repairllama_pad_id = repairllama_tokenizer.pad_token_id

#         required_columns = ['buggy_code', 'fixed_code', 'gpt_explanation']
#         if not all(col in self.data.columns for col in required_columns):
#             raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

#         if (self.data[['buggy_code', 'fixed_code', 'gpt_explanation']].isnull().any().any()):
#             raise ValueError(f"Dataframe contains 'null' values")
        
#     def __len__(self):
#         return len(self.data)
    
#     def __get_padding__(self, ids, pad_id, max_len):
#         padding_len = max_len - ids.shape[0]
#         if padding_len > 0:
#             ids = torch.cat((ids, torch.full((padding_len,), pad_id, dtype=torch.int64)))
#         elif padding_len<0:
#             ids = ids[: max_len]
#         return ids


#     def __getitem__(self, index):
#         try:
#             IGNORE_INDEX = -100
#             row = self.data.iloc[index]

#             # print(f"Row type: {type(row)}, Value: {repr(row)}", file=sys.stderr)

#             buggy_code = row['buggy_code']
#             fixed_code = row['fixed_code']
#             explanation = row['gpt_explanation']

#             # print(f"Type of buggy_code: {type(buggy_code)}, Value: {repr(buggy_code)}", file=sys.stderr)

#             prompt = PROMPT_DICT["prompt_input"].format_map({"buggy_code": buggy_code})
#             example= prompt + explanation

#             repairllama_prompt = buggy_code+ "\n // Fixed Code: \n"+ fixed_code


#             repairllama_encoding = self.repairllama_tokenizer.encode_plus(
#                 repairllama_prompt,
#                 max_length=self.repairllama_max_input_len,
#                 padding='max_length',
#                 truncation=True,
#                 return_tensors='pt'
#             )
#             repairllama_input_ids = repairllama_encoding['input_ids'].squeeze(0)  # [max_len]


#             prompt_encoding = self.llama_tokenizer.encode_plus(
#                 prompt, 
#                 padding="max_length", 
#                 max_length=self.llama_max_input_len, 
#                 truncation=True,
#                 return_tensors='pt'
#             )
#             prompt = prompt_encoding['input_ids'].squeeze(0)

#             example_encoding = self.llama_tokenizer.encode_plus(
#                 example,
#                 padding='max_length',
#                 max_length=self.llama_max_input_len,
#                 truncation=True,
#                 return_tensors='pt'
#             )
#             example = example_encoding['input_ids'].squeeze(0)
#             # example = self.llama_tokenizer.encode(example, padding="max_length", max_length=1024, truncation=True)
#             # example.append(self.llama_tokenizer.eos_token_id)
#             example = torch.cat([example, torch.tensor([self.llama_tokenizer.eos_token_id])])

#             # example = torch.tensor(
#             #     example, dtype=torch.int64
#             # )
#             labels = copy.deepcopy(example)
#             labels[: len(prompt)] = -1
#             example_mask = example.ge(0)
#             label_mask = labels.ge(0)
#             example[~example_mask] = 0
#             labels[~label_mask] = IGNORE_INDEX

#             return {
#                 "llama_input_ids": example.tolist(),
#                 "llama_label_ids": labels.tolist(),
#                 "llama_attention_mask":example_mask.tolist(),
#                 "repairllama_input_ids": repairllama_input_ids,
#             }
        
#         except Exception as e:
#             # Catch and log any exceptions
#             print(f"Error processing index {index}: {e}")
#             raise 


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
        self.data = pd.read_csv(dataframe_path, nrows=1000)
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


    # def __getitem__(self, idx):
    #     row = self.data.iloc[idx]
    #     buggy = str(row['buggy_code'])
    #     fixed = str(row['fixed_code'])
    #     explanation = str(row['gpt_explanation'])

    #     # --- RepairLlama side (just code + fixed) ---
    #     repair_text = buggy + "\n// Fixed Code:\n" + fixed + self.repair_tok.eos_token
    #     repair_enc = self.repair_tok(
    #         repair_text,
    #         padding="max_length",
    #         max_length=self.repair_max,
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     repair_input_ids = repair_enc.input_ids.squeeze(0)            # [1024]
    #     repairllama_mask = repair_enc.attention_mask.squeeze(0)      # [1024]

    #     # --- LLaMA side (prompt + explanation) ---
    #     prompt_text = PROMPT_DICT["prompt_input"].format(buggy_code=buggy)
    #     full_text   = prompt_text + explanation + self.llama_tok.eos_token

    #     llama_enc = self.llama_tok(
    #         full_text,
    #         padding="max_length",
    #         max_length=self.llama_max,
    #         truncation=True,
    #         return_tensors="pt",
    #     )

    #     llama_input_ids  = llama_enc.input_ids.squeeze(0)            # [1024]
    #     # right after constructing llama_input_ids:
    #     max_id = llama_input_ids.max().item()
    #     min_id = llama_input_ids.min().item()
    #     # print(f"[DEBUG] llama IDs in [{min_id}..{max_id}], vocab_size={len(self.llama_tok.get_vocab())}")
    #     assert max_id < len(self.llama_tok.get_vocab()), (
    #         f"Token ID {max_id} >= vocab_size {len(self.llama_tok.get_vocab())}"
    #     )

    #     llama_mask = llama_enc.attention_mask.squeeze(0)       # [1024]

    #     # --- build labels: mask out the prompt portion ---
    #     # 2) Separately encode just the explanation+EOS (no prompt, no padding)
    #     expl_enc = self.llama_tok(
    #         explanation + self.llama_tok.eos_token,
    #         padding=False,
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     expl_len = expl_enc.input_ids.size(1)   

    #     # print("expl len: _____", expl_len)
    #     llama_labels = llama_input_ids.clone()
    #     llama_labels[:-expl_len] = -100  # ignore prompt tokens

    #     return repair_input_ids, repairllama_mask, llama_input_ids, llama_labels, llama_mask
