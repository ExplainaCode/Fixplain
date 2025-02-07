import os
import argparse
from .codellama.model import ModelArgs, Transformer
import json
from .codellama.tokenizer import Tokenizer
import torch
from pathlib import Path
from .adapter_utils import sample_top_p
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def load_codellama_fsdp(rank, world_size, codellama_ckpt_dir, max_seq_len, max_batch_size, codellama_tokenizer, w_lora, lora_rank):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    with open(os.path.join(codellama_ckpt_dir, "params.json"), 'r') as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, 
        w_lora=w_lora, lora_rank=lora_rank,
        **params
    )
    tokenizer = Tokenizer(model_path=codellama_tokenizer)
    model_args.vocab_size = tokenizer.n_words

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    codellama = Transformer(model_args).to("cpu").half()

    # Load checkpoint
    ckpts = sorted(Path(codellama_ckpt_dir).glob("*.pth"))
    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        codellama.load_state_dict(ckpt, strict=False)
    codellama.half()

    # Wrap with FSDP instead of DDP
    codellama = FSDP(codellama, device_id=rank)

    return codellama, tokenizer

def load_codellama_ddp(rank, world_size, codellama_ckpt_dir, max_seq_len, max_batch_size, codellama_tokenizer, w_lora, lora_rank):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    with open(os.path.join(codellama_ckpt_dir, "params.json"), 'r') as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, 
        w_lora=w_lora, lora_rank=lora_rank,
        **params
    )
    tokenizer = Tokenizer(model_path=codellama_tokenizer)
    model_args.vocab_size = tokenizer.n_words
    
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    codellama = Transformer(model_args).to("cpu").half()

    # Load checkpoint
    ckpts = sorted(Path(codellama_ckpt_dir).glob("*.pth"))
    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        codellama.load_state_dict(ckpt, strict=False)
    codellama.half()

    codellama = DDP(codellama, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    return codellama, tokenizer


def load_codellama(codellama_ckpt_dir, max_seq_len, max_batch_size, codellama_tokenizer, w_lora, lora_rank):
    with open(os.path.join(codellama_ckpt_dir, "params.json"), 'r') as f:
        params = json.loads(f.read())
    
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, 
        w_lora=w_lora, lora_rank=lora_rank,
        **params
    )
    tokenizer = Tokenizer(model_path=codellama_tokenizer)
    tokenizer.pad_id = tokenizer.eos_id
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    codellama = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # codellama = codellama.to("cuda")

    # Print data type of model parameters
    # for name, param in codellama.named_parameters():
    #     print(f"Parameter: {name}, dtype: {param.dtype}")
    ckpts = sorted(Path(codellama_ckpt_dir).glob("*.pth"))
    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        missing_keys, unexpected_keys = codellama.load_state_dict(ckpt, strict=False)

        # print(f"Checkpoint: {ckpt_path}")
        # print("Missing Keys (not updated):", missing_keys)
        # print("Unexpected Keys (not in model):", unexpected_keys)
        # print("-" * 50)

    return codellama, tokenizer

def load_codellma_tuned(codellama, codellama_trained_weight_dir):
    ckpts = sorted(Path(codellama_trained_weight_dir).glob("*.pth"))
    ckpt_path = ckpts[-1]

    ckpt = torch.load(ckpt_path, map_location="cpu") # This ckeckpoint contains other parameters as well
    ckpt = ckpt["model"] 
    missing_keys, unexpected_keys = codellama.load_state_dict(ckpt, strict=False)

    # print("____________________in trained weights loading___________________")
    # print(f"Checkpoint: {ckpt_path}")
    # print("Missing Keys (not updated):", missing_keys)
    # print("Unexpected Keys (not in model):", unexpected_keys)
    # print("-" * 50)
    # print(ckpt.keys())
    return codellama

@torch.inference_mode()
def forward_inference(codellama, codellama_input_ids, codellama_start_pos:int, adapter=False):

    codellama_input_ids=codellama_input_ids.to(device)

    _bsz, codellama_seqlen = codellama_input_ids.shape
    codellama_h = codellama.tok_embeddings(codellama_input_ids)
    codellama_freq_cis = codellama.freqs_cis.to(codellama_h.device)
    codellama_freq_cis = codellama_freq_cis[:codellama_seqlen]
    codellama_mask = None
    codellama_mask = torch.full((1, 1, codellama_seqlen, codellama_seqlen), float("-inf"), device=codellama_h.device)
    codellama_mask = torch.triu(codellama_mask, diagonal=codellama_start_pos + 1).type_as(codellama_h)

    n_layers = codellama.config['num_hidden_layers']


    for i in range(n_layers):
        # dynamic_adapter  = attention_hooks_data[i].get('input') # Hooked input to the respective repairllama layer
        dynamic_adapter = torch.randn(1, codellama_seqlen, 4096, dtype=torch.float16, device='cuda')
        codellama_h = codellama.layers[i](codellama_h, codellama_start_pos, codellama_freq_cis, codellama_mask, dynamic_adapter)


    codellama_h = codellama.norm(codellama_h)
    codellama_output = codellama.output(codellama_h[:,-1, :])
 
    return codellama_output.float() if codellama_output is not None else None

@torch.inference_mode()
def generate(codellama, codellama_tokenizer, codellama_input_ids=None,
                max_gen_len: int=256, max_codellama_gen_len:int=125, temperature: float=0.1,
                top_p: float=0.75):
    bsz = 1# len(codellama_input_ids)
    # codellama_input_ids = ["""def fibonacci_iter(n):
    # a, b = 0, 1
    # for _ in range(n):"""]
    if codellama_input_ids==None:
        codellama_input_ids = [
            torch.full((1, 1), fill_value=codellama_tokenizer.pad_id, dtype=torch.long) #  torch.full((1, seq_len), fill_value=0, dtype=torch.long) 
            for _ in range(bsz)
        ]
        print("_______________")
        print(codellama_input_ids, codellama_input_ids[0].shape)
    
    # is this need to be checked. because batch sizes of both inputs are equal and both use same model. hece comment down and 
    # create a single params. check this
    # repairllama_params = self.repairllama.params
    # codellama_params = self.codellama.params
    # assert bsz <= repairllama_params.max_batch_size, (bsz, repairllama_params.max_batch_size)
    # assert bsz <= codellama_params.max_batch_size, (bsz, codellama_params.max_batch_size)

    # Replaced with params,
    params = codellama.params
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    if isinstance(codellama_input_ids[0], str):
        # This has custom tokenizer encode in codellama directory
        codellama_input_ids = [codellama_tokenizer.encode(x, bos=True, eos=False) for x in codellama_input_ids]
    
    #Clipplig to max_seq_len
    # Convert list of tensors into a single tensor
    codellama_input_ids = torch.stack(codellama_input_ids)
    codellama_input_ids = codellama_input_ids[:, :, :params.max_seq_len]

    min_codellama_prompt_size = min([len(t[0]) for t in codellama_input_ids])
    max_codellama_prompt_size = max([len(t[0]) for t in codellama_input_ids])

    # max_codellama_gen_len = max_gen_len # max_codellama_gen_len should be taken from the parameters, for the testing it is equal to the max_gen_len (in repairllama)
    total_codellama_len = min(params.max_seq_len, max_codellama_gen_len + max_codellama_prompt_size) # instead of generic params.max_seq_len consider using specific to codellama & max_gen_len for codellama text.
    codellama_tokens = torch.full((bsz, total_codellama_len), codellama_tokenizer.pad_id).cuda().long() # 0 used instead of self.codellama_tokenizer.pad_id for testing

    for k, t in enumerate(codellama_input_ids):
        if total_codellama_len <=len(t[0]):
            codellama_tokens[k, : total_codellama_len] = torch.tensor(t).cuda().long() # cuda
        else:
            codellama_tokens[k, : len(t[0])] = torch.tensor(t).cuda().long() # cuda

    input_codellama_text_mask = codellama_tokens != codellama_tokenizer.pad_id # o used instead of self.codellama_tokenizer.pad_id for testing (#important)
    codellama_start_pos = min_codellama_prompt_size
    # assert total_repairllama_len >= total_codellama_len

    prev_pos = 0
    codellama_pre_pos = 0

    for codellama_cur_pos in range(codellama_start_pos, total_codellama_len):
        with torch.cuda.amp.autocast():
            codellama_logits = forward_inference(codellama, codellama_tokens[:, codellama_pre_pos:codellama_cur_pos], codellama_pre_pos, adapter=True)

        if temperature > 0:
            probs = torch.softmax(codellama_logits / temperature, dim=-1)
            next_codellama_token = sample_top_p(probs, top_p)
        else:
            next_codellama_token = torch.argmax(codellama_logits, dim=-1)
        next_codellama_token = next_codellama_token.reshape(-1)
        # print("codellama_cur_pos: ", codellama_cur_pos)
        # print("codellama_tokens shape: ", codellama_tokens.shape)
        next_codellama_token = torch.where(
            input_codellama_text_mask[:, codellama_cur_pos], codellama_tokens[:, codellama_cur_pos], next_codellama_token
        )
        codellama_tokens[:, codellama_cur_pos] = next_codellama_token
        codellama_pre_pos=codellama_cur_pos

    
    codellama_decoded = []
    for i, t in enumerate(codellama_tokens.tolist()):

        # cut to max gen len
        t = t[len(codellama_input_ids[i]): len(codellama_input_ids[i]) + max_gen_len]
        # cut to eos tok if any
        try:
            t = t[: t.index(codellama_tokenizer.eos_id)]
        except ValueError:
            pass
        codellama_decoded.append(codellama_tokenizer.decode(t))

    return codellama_decoded

def main(args):
    codellama, tokenizer = load_codellama(
                                          args.codellama_ckpt_dir,
                                          args.max_seq_len, args.max_batch_size,
                                          args.codellama_tokenizer_path,
                                          False, 16)
    # codellama = load_codellma_tuned(codellama, args.codellama_trained_weight_dir)
    codellama_decoded = generate(codellama, tokenizer, None)
    print(codellama_decoded)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass configuration paths.")
    parser.add_argument("--codellama_ckpt_dir", type=str, required=True, help="Path to CodeLlama checkpoint directory")
    parser.add_argument("--codellama_tokenizer_path", type=str, required=True, help="Path to CodeLlama tokenizer")
    parser.add_argument("--codellama_trained_weight_dir", type=str, required=True, help="Path to trained weights")
    parser.add_argument("--repairllama_input_pth", type=str, required=True, help="RepairLLama input for forward inference")
    parser.add_argument("--codellama_input_pth", type=str, required=False, help="CodeLLAma input for forward inference")
    parser.add_argument("--repairllama_model_dir", type=str, required=False, help="Path to RepairLlama model directory")
    parser.add_argument("--repairllama_lora_dir", type=str, required=False, help="Path to RepairLlama LoRA directory")
    parser.add_argument("--max_batch_size", type=int, required=True, help="Max batch size")
    parser.add_argument('--max_seq_len', default=512, type=int, help='max number of input words')
    parser.add_argument('--w_lora', default=False, type=bool)
    parser.add_argument('--lora_rank', default=16, type=int, help='This only apply if the w_lora parameter is "True"')
    # parser.add_argument("--rank", type=int, default=int(os.environ["RANK"]), help="Process rank")
    # parser.add_argument("--world_size", type=int, default=int(os.environ["WORLD_SIZE"]), help="Total number of processes")
    
    args = parser.parse_args()
    main(args)
