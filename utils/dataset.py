import torch
from torch.utils.data import Dataset
# import llama.utils
# from codellama.tokenizer import Tokenizer
import copy
import pandas as pd
from dataclasses import dataclass
# from ..adapter import LLamaAdapter

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nThere is a buggy code provided and fixed code embeddings come through intermediate layers. write an explanation explaining bug and fix\n\n### Input:\n{buggy_code}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class FinetuneDataset(Dataset):
    def __init__(self, llama_tokenizer, repairllama_tokenizer, dataframe_path:str, phase='train'):# model:LLamaAdapter
        print(f"read dataset  from {dataframe_path}")
        self.data = pd.read_csv(dataframe_path)  # Load DataFrame from CSV file assumed have buggy_code, fixed_code and explanation columns
        self.llama_tokenizer = llama_tokenizer
        self.repairllama_tokenizer = repairllama_tokenizer
        self.repairllama_max_input_len = 1024 # model.llama_max_seq_len
        self.llama_max_input_len = 1024 # model.llama_max_seq_len
        self.llama_pad_id = llama_tokenizer.pad_token_id#model.llama_tokenizer.pad_id
        self.phase=phase
        # self.repairllama_pad_id = repairllama_tokenizer.pad_token_id

        required_columns = ['buggy_code', 'fixed_code', 'gpt_explanation']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

        if (self.data[['buggy_code', 'fixed_code', 'gpt_explanation']].isnull().any().any()):
            raise ValueError(f"Dataframe contains 'null' values")

    def __len__(self):
        return len(self.data)
    
    def __get_padding__(self, ids, pad_id, max_len):
        padding_len = max_len - ids.shape[0]
        if padding_len > 0:
            ids = torch.cat((ids, torch.full((padding_len,), pad_id, dtype=torch.int64)))
        elif padding_len<0:
            ids = ids[: max_len]
        return ids


    def __getitem__(self, index):
        try:
            IGNORE_INDEX = -100
            row = self.data.iloc[index]

            
            buggy_code = row['buggy_code']
            fixed_code = row['fixed_code']
            explanation = row['gpt_explanation']

            print(buggy_code)
            print("___________-")
            # print(f"Type of buggy_code: {type(buggy_code)}, Value: {buggy_code}")

            prompt = PROMPT_DICT["prompt_input"].format_map({"buggy_code": buggy_code})
            example= prompt + explanation

            repairllama_prompt = buggy_code+ "\n // Fixed Code: \n"+ fixed_code
            repairllama_encoding = self.repairllama_tokenizer.encode_plus(
                repairllama_prompt,
                max_length=self.repairllama_max_input_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            repairllama_input_ids = repairllama_encoding['input_ids'].squeeze(0)  # [max_len]


            prompt = torch.tensor(
                self.llama_tokenizer.encode(prompt), dtype=torch.int64
            )
            example = self.tokenizer.encode(example)
            example.append(self.tokenizer.eos_token_id)
            example = torch.tensor(
                example, dtype=torch.int64
            )
            labels = copy.deepcopy(example)
            labels[: len(prompt)] = -1
            example_mask = example.ge(0)
            label_mask = labels.ge(0)
            example[~example_mask] = 0
            labels[~label_mask] = IGNORE_INDEX

            return {
                "llama_input_ids": example.tolist(),
                "llama_label_ids": labels.tolist(),
                "llama_attention_mask":example_mask.tolist(),
                "repairllama_input_ids": repairllama_input_ids,
            }
        
        except Exception as e:
            # Catch and log any exceptions
            print(f"Error processing index {index}: {e}")
            raise 
