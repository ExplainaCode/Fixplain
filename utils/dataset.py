import torch
from torch.utils.data import Dataset
# import llama.utils
from codellama.tokenizer import Tokenizer
import copy
import pandas as pd
from dataclasses import dataclass
from transformers import (
AutoTokenizer,
)

@dataclass
class DatasetArgs:
    repairllama_max_input_len: int = 1024
    repairllama_max_output_len: int = 512 # no need for training 
    codellama_max_input_len: int = 256
    codellama_max_output_len: int = 256 # no need for training

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
        self.repairllama_pad_id = repairllama_tokenizer.pad_token_id

        required_columns = ['buggy_code', 'fixed_code', 'gpt_explanation']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")
        
        if (self.data[['buggy_code', 'gpt_explanation']].isnull().any().any()): # Remoed fixed_code_column from the list of null check
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
            fixed_code = row['fixed_code']
            explanation = row['gpt_explanation']
        
            repairllama_input_ids =  torch.flatten(self.repairllama_tokenizer.encode(buggy_code, return_tensors='pt'))
            repairllama_label_ids = torch.flatten(self.repairllama_tokenizer.encode(fixed_code, return_tensors='pt'))
            codellama_input_ids = torch.tensor(self.codellama_tokenizer.encode(explanation, bos=True, eos=False))

            repairllama_input_ids = self.__get_padding__(repairllama_input_ids, self.repairllama_pad_id, self.repairllama_max_input_len)
            repairllama_label_ids = self.__get_padding__(repairllama_label_ids, self.repairllama_pad_id, self.repairllama_max_output_len)
            codellama_input_ids = self.__get_padding__(codellama_input_ids, self.codellama_pad_id, self.codellama_max_input_len)

            codellama_label_ids = copy.deepcopy(codellama_input_ids)
            codellama_input_ids_mask  = codellama_input_ids.ge(0) # just for keep functions work for now - no need !
            # codellama_label_mask = codellama_label_ids.ge(0)
            # codellama_input_ids[~codellama_input_ids_mask] = 0
            # codellama_label_ids[~codellama_label_mask] = 0
            # codellama_label_mask = codellama_label_mask.float()
            # codellama_input_ids_mask = codellama_input_ids_mask.float()

            # if (index==0):
            #     print("repairllama_pad_id: ", self.repairllama_pad_id)
            #     print("codellama_pad_id: ", self.codellama_pad_id)
            #     print("repairllama_input_ids: ",repairllama_input_ids, repairllama_input_ids.shape)
            #     print("codellama_input_ids: ", codellama_input_ids, codellama_input_ids.shape)

            return repairllama_input_ids, repairllama_label_ids, codellama_input_ids, codellama_label_ids, codellama_input_ids_mask
        
        except Exception as e:
            # Catch and log any exceptions
            print(f"Error processing index {index}: {e}")
            print(f"fixed code type: {type(fixed_code)}, value: {fixed_code}")
            raise 
