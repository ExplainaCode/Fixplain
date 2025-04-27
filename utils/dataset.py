import torch
from torch.utils.data import Dataset
# import llama.utils
# from codellama.tokenizer import Tokenizer
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
    llama_max_input_len: int = 256
    llama_max_output_len: int = 256 # no need for training

    dataframe_path :str = ""


class FinetuneDataset(Dataset):
    def __init__(self, llama_tokenizer, repairllama_tokenizer, args:DatasetArgs):
        print(f"read dataset  from {args.dataframe_path}")
        self.data = pd.read_csv(args.dataframe_path)  # Load DataFrame from CSV file
        self.llama_tokenizer = llama_tokenizer
        self.repairllama_tokenizer = repairllama_tokenizer
        self.repairllama_max_input_len = args.repairllama_max_input_len
        self.repairllama_max_output_len = args.repairllama_max_output_len
        self.llama_max_input_len = args.llama_max_input_len
        self.llama_pad_id = llama_tokenizer.pad_id
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
            messages = row['messages']
            formatted_chat = self.llama_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
                # Tokenize entire sequence
            tokenized = self.llama_tokenizer(
                formatted_chat,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # Create labels mask
            labels = tokenized["input_ids"].clone()
            # Find where assistant response starts using special token
            response_start = torch.where(
                tokenized["input_ids"] == self.tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>")[0]
            )[1][0]

            labels[:, :response_start+1] = -100  # +1 to mask the header token itself
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "labels": labels.squeeze(0)
            }
        
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

            # repairllama_label_encoding = self.repairllama_tokenizer.encode_plus(
            #     fixed_code,
            #     max_length=self.repairllama_max_output_len,
            #     padding='max_length',
            #     truncation=True,
            #     return_tensors='pt'
            # )
            # repairllama_label_ids = repairllama_label_encoding['input_ids'].squeeze(0)
            # repairllama_input_ids =  torch.flatten(self.repairllama_tokenizer.encode(buggy_code, return_tensors='pt'))
            # repairllama_label_ids = torch.flatten(self.repairllama_tokenizer.encode(fixed_code, return_tensors='pt'))
            llama_input_ids = torch.tensor(self.llama_tokenizer.encode(explanation, bos=True, eos=False))

            # repairllama_input_ids = self.__get_padding__(repairllama_input_ids, self.repairllama_pad_id, self.repairllama_max_input_len)
            # repairllama_label_ids = self.__get_padding__(repairllama_label_ids, self.repairllama_pad_id, self.repairllama_max_output_len)
            llama_input_ids = self.__get_padding__(llama_input_ids, self.llama_pad_id, self.llama_max_input_len)

            llama_label_ids = copy.deepcopy(llama_input_ids)
            llama_input_ids_mask  = llama_input_ids.ge(0) # just for keep functions work for now - no need !
            # codellama_label_mask = llama_label_ids.ge(0)
            # llama_input_ids[~llama_input_ids_mask] = 0
            # llama_label_ids[~codellama_label_mask] = 0
            # codellama_label_mask = codellama_label_mask.float()
            # llama_input_ids_mask = llama_input_ids_mask.float()

            return repairllama_input_ids, llama_input_ids, llama_label_ids, llama_input_ids_mask
        
        except Exception as e:
            # Catch and log any exceptions
            print(f"Error processing index {index}: {e}")
            raise 
