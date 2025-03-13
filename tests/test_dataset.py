import torch
from transformers import AutoTokenizer
from ..utils.dataset import DatasetArgs, FinetuneDataset
# from ..codellama.tokenizer import Tokenizer
from ..codellama.tokenizer import Tokenizer

def __load_llama_tokenizer__():
    llama_tokenizer_pth = "tests/test_data/Llama3.1-8B/tokenizer.model"
    llama_tokenizer = Tokenizer(model_path=llama_tokenizer_pth)
    llama_tokenizer.pad_id = llama_tokenizer.eos_id
    return llama_tokenizer

def __load_repairllama_tokenizer__():
    repairllama_model_dir = "tests/test_data/CodeLlama-7b-hf"
    repairllama_tokenizer = AutoTokenizer.from_pretrained(
        repairllama_model_dir, 
        trust_remote_code=True,
        padding_size='left'
    )
    repairllama_tokenizer.pad_token = repairllama_tokenizer.unk_token
    repairllama_tokenizer.pad_token_id = repairllama_tokenizer.unk_token_id
    return repairllama_tokenizer

def test_dataset_load():
    dataset_args = DatasetArgs()
    dataset_args.dataframe_path = "tests/test_data/combined_repairllama_explanations_100.csv"

    llama_tokenizer = __load_llama_tokenizer__()
    repairllama_tokenizer = __load_repairllama_tokenizer__()

    finetune_dataset = FinetuneDataset(
        codellama_tokenizer=llama_tokenizer,
        repairllama_tokenizer=repairllama_tokenizer,
        args=dataset_args
    )

    (repairllama_input_ids, 
    codellama_input_ids, 
    codellama_label_ids, 
    codellama_input_ids_mask)  = finetune_dataset.__getitem__(0)

    assert isinstance(repairllama_input_ids, torch.Tensor)
    assert isinstance(codellama_input_ids, torch.Tensor)
    assert isinstance(codellama_label_ids, torch.Tensor)
    assert isinstance(codellama_input_ids_mask, torch.Tensor)

    # print(codellama_input_ids)
    # print(codellama_label_ids)

    assert(repairllama_input_ids.shape==torch.Size([dataset_args.repairllama_max_input_len]))
    assert(codellama_input_ids.shape==torch.Size([dataset_args.codellama_max_input_len]))
    assert(codellama_label_ids.shape==torch.Size([dataset_args.codellama_max_output_len]))
    assert(codellama_input_ids_mask.shape==torch.Size([dataset_args.codellama_max_input_len]))
