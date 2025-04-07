import torch
from torch.utils.data import Dataset
import re
# import llama.utils
# from codellama.tokenizer import Tokenizer
import copy
import pandas as pd
from dataclasses import dataclass
from transformers import (
AutoTokenizer,
)

def compute_token_offsets(text, tokenizer, token_ids):
    """
    Compute approximate character offsets for each token.
    This uses tokenizer.decode([token]) for each token ID and then searches
    for the token string in the original text (starting at a running offset).
    """
    token_strs = [tokenizer.decode([tid]) for tid in token_ids]
    offsets = []
    current_offset = 0
    for ts in token_strs:
        # Find the next occurrence of the token substring in text
        start = text.find(ts, current_offset)
        if start == -1:
            start = current_offset  # fallback if not found
        end = start + len(ts)
        offsets.append((start, end))
        current_offset = end
    return offsets

def compute_weight_mask(text, offsets, default_weight=1.0, code_weight=2.0):
    """
    Compute a weight mask (list of floats) for each token, where tokens
    overlapping a code span (text enclosed in backticks, single or double quotes)
    get a higher weight.
    """
    # Regex pattern to match spans enclosed by backticks, single or double quotes.
    # It matches: `...` or '...' or "..."
    pattern = r"(`[^`]+`|'[^']+'|\"[^\"]+\")"
    code_spans = [(m.start(), m.end()) for m in re.finditer(pattern, text)]
    weights = []
    for (start, end) in offsets:
        weight = default_weight
        # Check if the token span overlaps any code span.
        for cs, ce in code_spans:
            if not (end <= cs or start >= ce):
                weight = code_weight
                break
        weights.append(weight)
    return weights

@dataclass
class DatasetArgs:
    repairllama_max_input_len: int = 1024
    repairllama_max_output_len: int = 512 # no need for training 
    codellama_max_input_len: int = 256
    codellama_max_output_len: int = 256 # no need for training
    default_loss_weight = 0.2
    code_loss_weight = 1.0

    dataframe_path :str = ""


class FinetuneDataset(Dataset):
    def __init__(self, codellama_tokenizer, repairllama_tokenizer, args:DatasetArgs):
        print(f"read dataset  from {args.dataframe_path}")
        self.data = pd.read_csv(args.dataframe_path)  # Load DataFrame from CSV file
        self.codellama_tokenizer = codellama_tokenizer
        self.repairllama_tokenizer = repairllama_tokenizer
        self.repairllama_max_input_len = args.repairllama_max_input_len
        self.repairllama_max_output_len = args.repairllama_max_output_len
        self.codellama_max_input_len = args.codellama_max_input_len
        self.codellama_pad_id = codellama_tokenizer.pad_id
        self.default_weight = args.default_loss_weight
        self.code_weight = args.code_loss_weight
        # self.repairllama_pad_id = repairllama_tokenizer.pad_token_id

        required_columns = ['buggy_code', 'fixed_code', 'gpt_explanation']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")
        
        # this is for testing since fixed code does  not matter in finetuning.
        self.data['fixed_code'] = self.data['fixed_code'].fillna(" ") # remoe this if want

        if (self.data[['buggy_code', 'fixed_code', 'gpt_explanation']].isnull().any().any()):
            raise ValueError(f"Dataframe contains 'null' values")
        
        # df_cleaned = self.data.dropna(subset=['buggy_code', 'fixed_code', 'gpt_explanation'])
        # self.data = df_cleaned.reset_index(drop=True)

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
            row = self.data.iloc[index]
            buggy_code = row['buggy_code']
            # fixed_code = row['fixed_code']
            explanation = row['gpt_explanation']

            repairllama_encoding = self.repairllama_tokenizer.encode_plus(
                buggy_code,
                max_length=self.repairllama_max_input_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            repairllama_input_ids = repairllama_encoding['input_ids'].squeeze(0)  # [max_len]
            codellama_input_ids = torch.tensor(self.codellama_tokenizer.encode(explanation, bos=True, eos=False))

            offsets = compute_token_offsets(explanation, self.codellama_tokenizer, codellama_input_ids)
            # Compute a weight mask: tokens overlapping code spans get higher weight.
            loss_weight_mask = compute_weight_mask(explanation, offsets, 
                                                   default_weight=self.default_weight, 
                                                   code_weight=self.code_weight)
            loss_weight_mask = torch.tensor(loss_weight_mask, dtype=torch.float)

            codellama_input_ids = self.__get_padding__(codellama_input_ids, self.codellama_pad_id, self.codellama_max_input_len)
            loss_weight_mask = self.__get_padding__(loss_weight_mask, 0.0, self.codellama_max_input_len)

            codellama_label_ids = copy.deepcopy(codellama_input_ids)

            return repairllama_input_ids, codellama_input_ids, codellama_label_ids, loss_weight_mask
        
        except Exception as e:
            # Catch and log any exceptions
            print(f"Error processing index {index}: {e}")
            raise 
