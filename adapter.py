import torch
import torch.nn as nn
import os
import json
from pathlib import Path

from .codellama.model import ModelArgs, Transformer
from .codellama.tokenizer import Tokenizer
from .adapter_utils import sample_top_p
from peft import PeftModel
from transformers import (
AutoTokenizer,
AutoModelForCausalLM,
GenerationConfig,
HfArgumentParser,
BitsAndBytesConfig,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# codellama_device = device

class LLamaAdapter(nn.Module):
    def __init__(self,
                 codellama_ckpt_dir, codellama_tokenizer,
                 repairllama_lora_dir='./repairllama-lora', repairllama_model_dir="codellama/CodeLlama-7b-hf",
                 max_seq_len=512, max_batch_size=1,
                 v_embed_dim=768, v_depth=8,
                 v_num_heads=16, v_mlp_ratio=4.0,
                 query_len=10, query_layer=31,
                 w_bias=False, 
                 w_lora=False, lora_rank=16, 
                 w_new_gate=False,
                 phase="finetune",):
        super().__init__()
        self.attention_hooks_data = {} 
        self.codellama, self.codellama_tokenizer = self._load_codellama(
            codellama_ckpt_dir, max_seq_len, 
            max_batch_size, codellama_tokenizer)
        self.repairllama, self.repairllama_tokenizer = self._load_repairllama(
            repairllama_model_dir, repairllama_lora_dir,
            register_Attention_hooks=True)

    def _load_codellama(self, codellama_ckpt_dir, max_seq_len, max_batch_size, codellama_tokenizer):
        with open(os.path.join(codellama_ckpt_dir, "params.json"), 'r') as f:
            params = json.loads(f.read())
        
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        )
        tokenizer = Tokenizer(model_path=codellama_tokenizer)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        # torch.set_default_tensor_type(torch.FloatTensor) # load to cpu
        codellama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        ckpts = sorted(Path(codellama_ckpt_dir).glob("*.pth"))
        for ckpt in ckpts:
            ckpt = torch.load(ckpt, map_location="cpu")
            codellama.load_state_dict(ckpt, strict=False)

        return codellama, tokenizer 


    def _load_repairllama(self, repairllama_model_dir, repairllama_lora_dir, register_Attention_hooks=True):
        tokenizer = AutoTokenizer.from_pretrained(repairllama_model_dir, trust_remote_code=True)

        repairllama = AutoModelForCausalLM.from_pretrained(
            repairllama_model_dir,
            torch_dtype=torch.float16,
            # load_in_8bit=True, # commented initially
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            ),
        )

        repairllama = PeftModel.from_pretrained(
            repairllama,
            repairllama_lora_dir,
            torch_dtype=torch.float16,
        )
        repairllama.config.pad_token = tokenizer.pad_token = tokenizer.unk_token 

        if register_Attention_hooks:
            """
            Registers hooks on all LlamaSdpaAttention modules.
            """
            layer_id = 0
            for layer in repairllama.model.model.layers:
                # attention_layer = layer.self_attn
                layer.layer_id = layer_id  # Tag the layer with an ID
                layer.register_forward_hook(self._hook_fn)
                layer_id += 1

        return repairllama, tokenizer

    def _hook_fn(self, module, input, output):
        """
        Hook function to capture inputs of attention layers.
        """
        layer_id = module.layer_id
        print("inside hook_fn layer_id: ",layer_id)
        print("input inside hook_fn", input)
        print("Output inside hook fn: ", output)

        self.attention_hooks_data[layer_id] = {
            "input": tuple(inp.detach() for inp in input),
        }
    
    def forward(self, repairllama_input_ids, codellama_input_ids, 
                repairllama_labels, codellama_labels):
        assert repairllama_input_ids.shape[0]==codellama_input_ids.shape[0] # batch_size should be equal
        repairllama_input_ids=repairllama_input_ids.to(device)
        codellama_input_ids=codellama_input_ids.to(device)
        # RepairLLama configuration before forward pass
        _bsz, repairllama_seqlen = repairllama_input_ids.shape

        repairllama_h = self.repairllama.tok_embeddings(repairllama_input_ids) # assuming toke_embedding is in reapirllama
        # repairllama_freqs_cis = self.repairllama.freqs_cis.to(repairllama_h.device) 
        # repairllama_freqs_cis = repairllama_freqs_cis[:repairllama_seqlen]
        repairllama_position_ids = torch.arange(repairllama_seqlen, dtype=torch.long, device=repairllama_input_ids.device).unsqueeze(0).expand(_bsz, -1)
        repairllama_mask = None
        repairllama_mask = torch.full((1, 1, repairllama_seqlen, repairllama_seqlen), float("-inf"), device=repairllama_h.device)
        repairllama_mask = torch.triu(repairllama_mask, diagonal=0 + 1).type_as(repairllama_h)

        # CodeLLama configuration before forward pass # This is redundent if works movw to a function or something...
        _bsz, codellama_seqlen = codellama_input_ids.shape
        codellama_h = self.codellama.tok_embeddings(codellama_input_ids)
        codellama_freq_cis = self.codellama.freqs_cis.to(codellama_h.device)
        codellama_freq_cis = codellama_freq_cis[:codellama_seqlen]
        codellama_mask = None
        codellama_mask = torch.full((1, 1, codellama_seqlen, codellama_seqlen), float("-inf"), device=codellama_h.device)
        codellama_mask = torch.triu(codellama_mask, diagonal=0 + 1).type_as(repairllama_h)

        assert self.repairllama.config.num_hidden_layers==self.codellama.config['num_hidden_layers']
        n_layers = self.repairllama.config.num_hidden_layers

        for i in range(n_layers):
            repairllama_h = self.repairllama.model.model.layers[i](hidden_states=repairllama_h, 
                                                       attention_mask=repairllama_mask, 
                                                       position_ids=repairllama_position_ids)
            assert(self.attention_hooks_data.get(i)!=None)
            dynamic_adaptor = self.attention_hooks_data[i].get('input')[0] # Hooked input to the respective repairllama layer
            codellama_h = self.codellama.layers[i](codellama_h, 0, codellama_freq_cis, codellama_mask, dynamic_adaptor)

        self.attention_hooks_data={} # Resetting can also be done in the above loop. 


        # Processing RepairLLama output
        repairllama_h = self.repairllama.norm(repairllama_h)
        repairllama_output = self.repairllama.output(repairllama_h)
        repairllama_output = repairllama_output[:, :-1, :]
        repairllama_labels = repairllama_labels[:, 1:]

        if repairllama_labels.sum() == 0:
            reapirllama_c_loss = repairllama_output.mean() * 0
        else:
            assert self.repairllama.vocab_size == 32000
            reapirllama_c_loss = self.criterion(repairllama_output.reshape(-1, self.repairllama.vocab_size), repairllama_labels.flatten())

        # Processing CodeLLama output
        codellama_h = self.codellama.norm(codellama_h)
        codellama_output = self.codellama.output(codellama_h)
        codellama_output = codellama_output[:, :-1, :]
        codellama_labels = codellama_labels[:, 1:]

        if codellama_labels.sum() ==0 :
            codellama_c_loss = codellama_output.mean() * 0
        else:
            assert self.codellama.vocab_size == 3200
            codellama_c_loss = self.criterian(codellama_output.reshape(-1, self.codellama.vocab_size), codellama_labels.flatten())

        return reapirllama_c_loss, codellama_c_loss
    
    @torch.inference_mode()
    def forward_inference(self, repairllama_input_ids, codellama_input_ids, start_pos:int):
        assert repairllama_input_ids.shape[0]==codellama_input_ids.shape[0] # batch_size should be equal

        repairllama_input_ids=repairllama_input_ids.to(device) #Decide whether this is the optimal position to move to the device #probably in training we can directly load to the device at once?
        codellama_input_ids=codellama_input_ids.to(device)
        # RepairLLama configuration before forward pass
        _bsz, repairllama_seqlen = repairllama_input_ids.shape

        repairllama_h = self.repairllama.model.model.embed_tokens(repairllama_input_ids) # apass through embedding layer
        # repairllama_freqs_cis = self.repairllama.freqs_cis.to(repairllama_h.device) 
        # repairllama_freqs_cis = repairllama_freqs_cis[:repairllama_seqlen]
        repairllama_position_ids = torch.arange(repairllama_seqlen, dtype=torch.long, device=repairllama_input_ids.device).unsqueeze(0).expand(_bsz, -1)
        repairllama_mask = None
        repairllama_mask = torch.full((1, 1, repairllama_seqlen, repairllama_seqlen), float("-inf"), device=repairllama_h.device)
        repairllama_mask = torch.triu(repairllama_mask, diagonal=0 + 1).type_as(repairllama_h)

        # CodeLLama configuration before forward pass # This is redundent if works movw to a function or something...
        _bsz, codellama_seqlen = codellama_input_ids.shape
        codellama_h = self.codellama.tok_embeddings(codellama_input_ids)
        codellama_freq_cis = self.codellama.freqs_cis.to(codellama_h.device)
        codellama_freq_cis = codellama_freq_cis[:codellama_seqlen]
        codellama_mask = None
        codellama_mask = torch.full((1, 1, codellama_seqlen, codellama_seqlen), float("-inf"), device=codellama_h.device)
        codellama_mask = torch.triu(codellama_mask, diagonal=0 + 1).type_as(repairllama_h)

        assert self.repairllama.config.num_hidden_layers==self.codellama.config['num_hidden_layers']
        n_layers = self.repairllama.config.num_hidden_layers

        for i in range(n_layers):
            print("repairllama_h :",repairllama_h)
            repairllama_h, *_ = self.repairllama.model.model.layers[i](
                                                repairllama_h, repairllama_mask, repairllama_position_ids
                                            )  # Do not pass as keyword arguments since hooks don't capture inputs.        
            assert(self.attention_hooks_data.get(i)!=None)
            print(self.attention_hooks_data)
            dynamic_adaptor = self.attention_hooks_data[i].get('input')[0] # Hooked input to the respective repairllama layer
            codellama_h = self.codellama.layers[i](codellama_h, start_pos, codellama_freq_cis, codellama_mask, dynamic_adaptor)

        self.attention_hooks_data={} # Resetting can also be done in the above loop. 


        # Processing RepairLLama output
        repairllama_h = self.repairllama.model.model.norm(repairllama_h[0])
        repairllama_output = self.repairllama.output(repairllama_h[: ,-1, :])

        # Processing CodeLLama output
        codellama_h = self.codellama.norm(codellama_h)
        codellama_output = self.codellama.output(codellama_h[:,-1, :])

        return repairllama_output.float(), codellama_output.float()
    
    @torch.inference_mode() #To be completed
    def generate(self, repairllama_input_ids, codellama_input_ids,
                 max_gen_len: int=256, temperature: float=0.1,
                 top_p: float=0.75):
        assert len(repairllama_input_ids)==len(codellama_input_ids)
        bsz = len(repairllama_input_ids)
        repairllama_params = self.repairllama.params
        codellama_params = self.codellama.params
        assert bsz <= repairllama_params.max_batch_size, (bsz, repairllama_params.max_batch_size)
        assert bsz <= codellama_params.max_batch_size, (bsz, codellama_params.max_batch_size)

        if isinstance(repairllama_input_ids[0], str): # if the inputs are given as strings instead of input_ids
            repairllama_input_ids = [self.repairllama_tokenizer.encode(x, bos=True, eos=False) for x in repairllama_input_ids]
        
        if isinstance(codellama_input_ids[0], str):
            codellama_input_ids = [self.codellama_tokenizer.encode(x, bos=True, eos=False) for x in codellama_input_ids]

        min_repairllama_prompt_size = min([len(t) for t in repairllama_input_ids])
        max_repairllama_prompt_size = max([len(t) for t in repairllama_input_ids])
        min_codellama_prompt_size = min([len(t) for t in codellama_input_ids])
        max_codellama_prompt_size = max([len(t) for t in repairllama_input_ids])

        total_repairllama_len = min(repairllama_params.max_seq_len, max_gen_len + max_repairllama_prompt_size)
        repairllama_tokens = torch.full((bsz, total_repairllama_len), self.repairllama_tokenizer.pad_id).cuda().long() # cuda --> cpu

        total_codellama_len = min(codellama_params.max_seq_len, max_gen_len + max_codellama_prompt_size)
        codellama_tokens = torch.full((bsz, total_codellama_len), self.codellama_tokenizer.pad_id).cuda().long() # cuda -->cpu changed by me

        for k, t in enumerate(repairllama_input_ids):
            repairllama_tokens[k, : len(t)] = torch.tensor(t).cuda().long() #cuda
        input_text_mask = repairllama_tokens != self.repairllama_tokenizer.pad_id
        start_pos = min_repairllama_prompt_size

        for k, t in enumerate(codellama_input_ids):
            codellama_tokens[k, : len(t)] = torch.tensor(t).cuda().long() # cuda

        prev_pos = 0
        for cur_pos in range(start_pos, total_repairllama_len):
            with torch.cuda.amp.autocast():
                logits = self.forward_inference(repairllama_tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], repairllama_tokens[:, cur_pos], next_token
            )
            repairllama_tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(repairllama_tokens.tolist()):

            # cut to max gen len
            t = t[len(repairllama_input_ids[i]): len(repairllama_input_ids[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded
