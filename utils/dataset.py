import torch
from torch.utils.data import Dataset
# import llama.utils
# from codellama.tokenizer import Tokenizer
import copy
import pandas as pd
from dataclasses import dataclass
from ..adapter import LLamaAdapter

class FinetuneDataset(Dataset):
    def __init__(self, model:LLamaAdapter, dataframe_path:str):
        print(f"read dataset  from {dataframe_path}")
        self.data = pd.read_csv(dataframe_path)  # Load DataFrame from CSV file
        self.llama_tokenizer = model.llama_tokenizer
        self.repairllama_tokenizer = model.repairllama_tokenizer
        self.repairllama_max_input_len = model.llama_max_seq_len
        self.llama_max_input_len = model.llama_max_seq_len
        self.llama_pad_id = model.llama_tokenizer.pad_id
        # self.repairllama_pad_id = repairllama_tokenizer.pad_token_id

        required_columns = ['buggy_code', 'fixed_code', 'gpt_explanation']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")
        
        # this is for testing since fixed code does  not matter in finetuning.
        self.data['fixed_code'] = self.data['fixed_code'].fillna(" ") # remoe this if want

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
            codellama_input_ids = self.__get_padding__(codellama_input_ids, self.codellama_pad_id, self.codellama_max_input_len)

            codellama_label_ids = copy.deepcopy(codellama_input_ids)
            codellama_input_ids_mask  = codellama_input_ids.ge(0) # just for keep functions work for now - no need !

            return repairllama_input_ids, codellama_input_ids, codellama_label_ids, codellama_input_ids_mask
        
        except Exception as e:
            # Catch and log any exceptions
            print(f"Error processing index {index}: {e}")
            raise 
