import torch
import yaml
from torch.utils.data import Dataset
import json
# import llama.utils
from codellama.tokenizer import Tokenizer
import copy
import pandas as pd
import random
from dataclasses import dataclass
from transformers import (
AutoTokenizer,
)

@dataclass
class DatasetArgs:
    repairllama_max_input_len = 1024
    repairllama_max_output_len = 512 # no need for training 
    codellama_max_input_len = 256
    codellama_max_output_len = 256 # no need for training

    dataframe_path :str = ""


class FinetuneDataset(Dataset):
    def __init__(self, codellama_tokenizer_path, repairllama_model_dir, args:DatasetArgs):
        print(f"read dataset  from {args.dataframe_path}")
        self.data = pd.read_csv(args.dataframe_path)  # Load DataFrame from CSV file
        self.codellama_tokenizer = Tokenizer(model_path=codellama_tokenizer_path)
        self.repairllama_tokenizer = AutoTokenizer.from_pretrained(repairllama_model_dir, trust_remote_code=True)
        self.repairllama_max_input_len = args.repairllama_max_input_len
        self.repairllama_max_output_len = args.repairllama_max_output_len
        self.codellama_max_input_len = args.codellama_max_input_len

        required_columns = ['buggy_code', 'fixed_code', 'gpt_explanation']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")
        
        if (self.data[['buggy_code', 'fixed_code', 'gpt_explanation']].isnull().any().any()):
            raise ValueError(f"Dataframe contains 'null' values")
        
        # df_cleaned = self.data.dropna(subset=['buggy_code', 'fixed_code', 'gpt_explanation'])
        # self.data = df_cleaned.reset_index(drop=True)

    def __len__(self):
        return len(self.data)
    
    def __get_padding__(self, ids, max_len, minus=0):
        padding_len = max_len - ids.shape[0]
        if padding_len > 0:
            ids = torch.cat((ids, torch.zeros(padding_len, dtype=torch.int64) - minus )) # for now padding is number 0, check with this with tokenizers.
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

            repairllama_input_ids = self.__get_padding__(repairllama_input_ids, self.repairllama_max_input_len, minus = 0)
            repairllama_label_ids = self.__get_padding__(repairllama_label_ids, self.repairllama_max_output_len, minus =0)
            codellama_input_ids = self.__get_padding__(codellama_input_ids, self.codellama_max_input_len, minus = 1)

            codellama_label_ids = copy.deepcopy(codellama_input_ids)
            codellama_input_ids_mask  = codellama_input_ids.ge(0)
            codellama_label_mask = codellama_label_ids.ge(0)
            codellama_input_ids[~codellama_input_ids_mask] = 0
            codellama_label_ids[~codellama_label_mask] = 0
            codellama_label_mask = codellama_label_mask.float()
            codellama_input_ids_mask = codellama_input_ids_mask.float()

            return repairllama_input_ids, repairllama_label_ids, codellama_input_ids, codellama_label_ids, codellama_input_ids_mask
        
        except Exception as e:
            # Catch and log any exceptions
            print(f"Error processing index {index}: {e}")
            print(f"fixed code type: {type(fixed_code)}, value: {fixed_code}")
            raise 


# class PretrainDataset(Dataset):
#     def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
#         print(f"read dataset config from {config_path}")
#         with open(config_path, 'r') as f:
#             self.config = yaml.load(f, Loader=yaml.FullLoader)
#         print("DATASET CONFIG:")
#         print(self.config)
#         images, captions = [], []
#         for meta_path in self.config['META']:
#             images_this_meta, captions_this_meta = [], []
#             for chunk in pd.read_csv(meta_path, sep='\t', lineterminator='\n', chunksize=10 ** 6):
#                 images_this_meta.extend(chunk['url'].tolist())
#                 captions_this_meta.extend(chunk['caption'].tolist())
#             print(f"{meta_path}: len {len(images_this_meta)}")
#             images.extend(images_this_meta)
#             captions.extend(captions_this_meta)

#         self.data_list = []
#         for x, y in zip(images, captions):
#             self.data_list.append({'url': x, 'caption': y})
#         print(f"total length: {len(self)}")
#         self.transform = transform
#         self.max_words = max_words
#         self.tokenizer = Tokenizer(model_path=tokenizer_path)

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, index):
#         sample = self.data_list[index]
#         image_path, caption = sample['url'], sample['caption']
#         if isinstance(caption, list):
#             caption = random.choice(caption)
#         caption = str(caption)

#         image = Image.open(image_path).convert('RGB')
#         image = self.transform(image)

#         format_instruction = "Generate caption of this image"
#         input1 = llama.utils.format_prompt(format_instruction, None)
#         input2 = input1 + caption

#         input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
#         input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
#         padding = self.max_words - input2.shape[0]
#         if padding > 0:
#             input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
#         elif padding < 0:
#             input2 = input2[:self.max_words]
#         labels = copy.deepcopy(input2)
#         labels[:len(input1)] = -1
#         input2_mask = input2.ge(0)
#         label_mask = labels.ge(0)
#         input2[~input2_mask] = 0
#         labels[~label_mask] = 0
#         input2_mask = input2_mask.float()
#         label_mask = label_mask.float()
#         return input2, labels, input2_mask, image