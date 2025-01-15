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
                 max_seq_len=512, max_batch_size=2,
                 w_bias=False, 
                 w_lora=False, lora_rank=16, 
                 w_new_gate=False,
                 phase="inference",):
        super().__init__()
        self.attention_hooks_data = {} 

        self.codellama, self.codellama_tokenizer = self._load_codellama(
            codellama_ckpt_dir, max_seq_len,
            max_batch_size, codellama_tokenizer,
            w_lora, lora_rank)
        self.repairllama, self.repairllama_tokenizer = self._load_repairllama(
            repairllama_model_dir, repairllama_lora_dir,
            register_Attention_hooks=True)
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.phase = phase
        self.set_trainale_params(self.phase)

    def _load_codellama(self, codellama_ckpt_dir, max_seq_len, max_batch_size, codellama_tokenizer, w_lora, lora_rank):
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
        self.attention_hooks_data[layer_id] = { # {0:{"input": (x, )}}
            # "input": tuple(inp.detach() for inp in input),
            "input": input[0].detach(),
        }

    def set_trainale_params(self, phase='inference'):
        for name, para in self.named_parameters():
            para.requires_grad = False

        if phase == 'finetune':
            target_keywords = ["lora", "adapter", "gate"]
            for name, para in self.named_parameters():
                if name.startswith("codellama"):
                    if any(keyword in name for keyword in target_keywords):
                        para.data = para.data.float()
                        para.requires_grad = True
                # print(name, para.requires_grad)    #debugging
        
        elif phase == 'inference':
            pass

        else:
            raise ValueError(f"Unknown model phase: {phase}")


    def forward(self, repairllama_input_ids, codellama_input_ids, 
                repairllama_labels, codellama_labels, repairllama_past_key_values=None):
        # assert repairllama_input_ids.shape[0]==codellama_input_ids.shape[0] # batch_size should be equal
        repairllama_input_ids=repairllama_input_ids.to(device)
        codellama_input_ids=codellama_input_ids.to(device)
        # RepairLLama configuration before forward pass
        _bsz, repairllama_seqlen = repairllama_input_ids.shape

        repairllama_h = self.repairllama.model.model.embed_tokens(repairllama_input_ids)
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

        # if repairllama_past_key_values is None:
        #     from transformers.cache_utils import DynamicCache
        #     repairllama_past_key_values = DynamicCache()

        # repairllama_past_key_values_len = repairllama_past_key_values.__len__()
        for i in range(n_layers):
            # if i < repairllama_past_key_values_len:
            #     past_key_values = repairllama_past_key_values.__getitem__(i)
            # else:
            #     past_key_values = None
            # if past_key_values:
            #     past_key_values = tuple(pkv.contiguous() for pkv in past_key_values)
            repairllama_h, *_ = self.repairllama.model.model.layers[i](
                                                repairllama_h.contiguous(), repairllama_mask.contiguous(), repairllama_position_ids.contiguous()
                                            )  # Do not pass as keyword arguments since hooks don't capture inputs.   
            assert(self.attention_hooks_data.get(i)!=None)
            dynamic_adapter = self.attention_hooks_data[i].get('input') # Hooked input to the respective repairllama layer
            codellama_h = self.codellama.layers[i](codellama_h, 0, codellama_freq_cis, codellama_mask, dynamic_adapter)
        self.attention_hooks_data={} # Resetting can also be done in the above loop. 


        # Processing RepairLLama output
        repairllama_h = self.repairllama.model.model.norm(repairllama_h) # Why do even need this line?
        repairllama_output = self.repairllama.model.lm_head(repairllama_h[:, -1, :]) # Why do even need this line?
        # repairllama_output = repairllama_output[:, :-1, :]
        # repairllama_labels = repairllama_labels[:, 1:]

        # if repairllama_labels.sum() == 0:
        #     reapirllama_c_loss = repairllama_output.mean() * 0
        # else:
        #     assert self.repairllama.vocab_size == 32000
        #     reapirllama_c_loss = self.criterion(repairllama_output.reshape(-1, self.repairllama.vocab_size), repairllama_labels.flatten())

        # Processing CodeLLama output

        codellama_h = self.codellama.norm(codellama_h)
        codellama_output = self.codellama.output(codellama_h)
        codellama_output = codellama_output[:, :-1, :]
        codellama_labels = codellama_labels[:, 1:]

        if codellama_labels.sum()==0 :
            codellama_c_loss = codellama_output.mean() * 0
        else:
            assert self.codellama.vocab_size == self.codellama_tokenizer.n_words #Do we need this line?, in load codellama this is set
            codellama_c_loss = self.criterian(codellama_output.reshape(-1, self.codellama.vocab_size), codellama_labels.flatten())

        return codellama_c_loss
    
    @torch.inference_mode()
    def forward_inference(self, repairllama_input_ids, codellama_input_ids, start_pos:int, repairllama_past_key_values=None, adapter=False):
        # assert repairllama_input_ids.shape[0]==codellama_input_ids.shape[0] # batch_size should be equal

        repairllama_input_ids=repairllama_input_ids.to(device) #Decide whether this is the optimal position to move to the device #probably in training we can directly load to the device at once?
        if adapter:
            codellama_input_ids=codellama_input_ids.to(device)
        # RepairLLama configuration before forward pass
        _bsz, repairllama_seqlen = repairllama_input_ids.shape

        repairllama_h = self.repairllama.model.model.embed_tokens(repairllama_input_ids) # apass through embedding layer

        # repairllama_freqs_cis = self.repairllama.freqs_cis.to(repairllama_h.device) 
        # repairllama_freqs_cis = repairllama_freqs_cis[:repairllama_seqlen]
        repairllama_position_ids = torch.arange(repairllama_seqlen, dtype=torch.long, device=repairllama_input_ids.device).unsqueeze(0).expand(_bsz, -1)
        repairllama_mask = None
        repairllama_mask = torch.full((1, 1, repairllama_seqlen, repairllama_seqlen), float("-inf"), device=repairllama_h.device)
        repairllama_mask = torch.triu(repairllama_mask, diagonal=0 + 1).type_as(repairllama_h) #this should change.

        if adapter:
            # CodeLLama configuration before forward pass # This is redundent if works movw to a function or something...
            _bsz, codellama_seqlen = codellama_input_ids.shape
            codellama_h = self.codellama.tok_embeddings(codellama_input_ids)
            codellama_freq_cis = self.codellama.freqs_cis.to(codellama_h.device)
            codellama_freq_cis = codellama_freq_cis[:codellama_seqlen]
            codellama_mask = None
            codellama_mask = torch.full((1, 1, codellama_seqlen, codellama_seqlen), float("-inf"), device=codellama_h.device)
            codellama_mask = torch.triu(codellama_mask, diagonal=0 + 1).type_as(repairllama_h) #This should change

        assert self.repairllama.config.num_hidden_layers==self.codellama.config['num_hidden_layers']
        n_layers = self.repairllama.config.num_hidden_layers

        if repairllama_past_key_values is None:
            from transformers.cache_utils import DynamicCache
            repairllama_past_key_values = DynamicCache()

        # print("________________________________________")
        # print("repairllama_h: ", repairllama_h.shape)
        # print("repairllama_mask: ", repairllama_mask.shape)
        # print("repairllama_position_ids: ", repairllama_position_ids)
        # print("repairllama_past_key_values: ", repairllama_past_key_values.__len__())

        repairllama_past_key_values_len = repairllama_past_key_values.__len__()
        for i in range(n_layers):
            if i < repairllama_past_key_values_len:
                past_key_values = repairllama_past_key_values.__getitem__(i)
            else:
                past_key_values = None
            if past_key_values:
                past_key_values = tuple(pkv.contiguous() for pkv in past_key_values)
            # print(repairllama_h)
            # print(repairllama_mask)
            # print(repairllama_position_ids)

            repairllama_h, next_repairllama_cache, *_ = self.repairllama.model.model.layers[i](
                                                repairllama_h.contiguous(), repairllama_mask.contiguous(), repairllama_position_ids.contiguous(), past_key_values, use_cache=True
                                            )  # Do not pass as keyword arguments since hooks don't capture inputs.        
            assert(self.attention_hooks_data.get(i)!=None)
            # print(self.attention_hooks_data)
            if adapter:
                dynamic_adapter = self.attention_hooks_data[i].get('input') # Hooked input to the respective repairllama layer
                codellama_h = self.codellama.layers[i](codellama_h, start_pos, codellama_freq_cis, codellama_mask, dynamic_adapter)

        # print(self.attention_hooks_data)   
        self.attention_hooks_data={} # Resetting can also be done in the above loop. 


        # Processing RepairLLama output
        repairllama_h  = self.repairllama.model.model.norm(repairllama_h)
        # repairllama_h, *_  = self.repairllama.model.model.rotary_emb(repairllama_h, position_ids=repairllama_position_ids)
        # print("repairllama shape 3: ", repairllama_h[:, -1, :].shape)
        repairllama_output = self.repairllama.model.lm_head(repairllama_h[:, -1, :])  # We assume that lm_lead accepts (batch_size, voc_size), not (batch_size, seq_len, voc_size) check this.

        if adapter:
            # Processing CodeLLama output
            codellama_h = self.codellama.norm(codellama_h)
            codellama_output = self.codellama.output(codellama_h[:,-1, :])
        else: 
            codellama_output = None

        return repairllama_output, codellama_output.float() if codellama_output is not None else None, next_repairllama_cache
    
    @torch.inference_mode() #To be completed
    def generate(self, repairllama_input_ids, codellama_input_ids=None,
                 max_gen_len: int=256, temperature: float=0.1,
                 top_p: float=0.75):
        bsz = len(repairllama_input_ids)
        if codellama_input_ids==None:
            codellama_input_ids = [
                torch.full((1, 1), fill_value=0, dtype=torch.long) #  torch.full((1, seq_len), fill_value=0, dtype=torch.long) 
                for _ in range(bsz)
            ]
        assert len(repairllama_input_ids)==len(codellama_input_ids) #batch sizes should be equal.
       
       # is this need to be checked. because batch sizes of both inputs are equal and both use same model. hece comment down and 
       # create a single params. check this
        # repairllama_params = self.repairllama.params
        # codellama_params = self.codellama.params
        # assert bsz <= repairllama_params.max_batch_size, (bsz, repairllama_params.max_batch_size)
        # assert bsz <= codellama_params.max_batch_size, (bsz, codellama_params.max_batch_size)

        # Replaced with params,
        params = self.codellama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        if isinstance(repairllama_input_ids[0], str): # if the inputs are given as strings instead of input_ids
             #This assumes list of pytorch tensors returns given enumerable (list) of input texts.
            repairllama_input_ids = [self.repairllama_tokenizer.encode(x, return_tensors='pt') for x in repairllama_input_ids]
        
        if isinstance(codellama_input_ids[0], str):
            # This has custom tokenizer encode in codellama directory
            codellama_input_ids = [self.codellama_tokenizer.encode(x, bos=True, eos=False) for x in codellama_input_ids]

        min_repairllama_prompt_size = min([len(t[0]) for t in repairllama_input_ids])
        max_repairllama_prompt_size = max([len(t[0]) for t in repairllama_input_ids])
        min_codellama_prompt_size = min([len(t[0]) for t in codellama_input_ids])
        max_codellama_prompt_size = max([len(t[0]) for t in codellama_input_ids])

        max_codellama_gen_len = max_gen_len # max_codellama_gen_len should be taken from the parameters, for the testing it is equal to the max_gen_len (in repairllama)
        total_repairllama_len = min(params.max_seq_len, max_gen_len + max_repairllama_prompt_size)
        repairllama_tokens = torch.full((bsz, total_repairllama_len), self.repairllama_tokenizer.pad_token_id).cuda().long()
        total_codellama_len = min(params.max_seq_len, max_codellama_gen_len + max_codellama_prompt_size) # instead of generic params.max_seq_len consider using specific to codellama & max_gen_len for codellama text.
        codellama_tokens = torch.full((bsz, total_codellama_len), 0).cuda().long() # 0 used instead of self.codellama_tokenizer.pad_id for testing

        for k, t in enumerate(repairllama_input_ids):
            repairllama_tokens[k, : len(t[0])] = torch.tensor(t).cuda().long()

        input_repairllama_text_mask = repairllama_tokens != self.repairllama_tokenizer.pad_token_id
        repairllama_start_pos = min_repairllama_prompt_size

        for k, t in enumerate(codellama_input_ids):
            codellama_tokens[k, : len(t[0])] = torch.tensor(t).cuda().long() # cuda

        input_codellama_text_mask = codellama_tokens != 0 # o used instead of self.codellama_tokenizer.pad_id for testing
        codellama_start_pos = min_codellama_prompt_size
        # assert total_repairllama_len >= total_codellama_len
        codellama_iter_start_pos = (total_repairllama_len - min_repairllama_prompt_size) - (total_codellama_len - min_codellama_prompt_size)
        if codellama_iter_start_pos < 0: 
            codellama_iter_start_pos = 0

        prev_pos = 0
        codellama_pre_pos = 0
        codellama_cur_pos=codellama_start_pos
        next_repairllama_cache = None
        for cur_pos in range(repairllama_start_pos, total_repairllama_len):
            with torch.cuda.amp.autocast():
                if cur_pos -repairllama_start_pos  <= codellama_iter_start_pos:
                    repairllama_output, _ , next_repairllama_cache = self.forward_inference(repairllama_tokens[:, prev_pos:cur_pos], None, codellama_pre_pos,repairllama_past_key_values=next_repairllama_cache, adapter=False)
                else:
                    repairllama_output, codellama_logits, next_repairllama_cache = self.forward_inference(repairllama_tokens[:, prev_pos:cur_pos], codellama_tokens[:, codellama_pre_pos:codellama_cur_pos], codellama_pre_pos, repairllama_past_key_values=next_repairllama_cache, adapter=True)
            # print("Repairllama logits: ", repairllama_logits, repairllama_logits.shape)
            # if temperature > 0:
            #     probs = torch.softmax(repairllama_logits / temperature, dim=-1)
            #     next_repairllama_token = sample_top_p(probs, top_p)
            # else:
            #     next_repairllama_token = torch.argmax(repairllama_logits, dim=-1)
            # print("Next_repairllama_token_before modification: ", next_repairllama_token)
            # print(next_repairllama_token.shape)
            # next_repairllama_token = repairllama_output.reshape(-1)
            next_repairllama_token = torch.argmax(repairllama_output, dim=-1) # samplelling is not used naive approach, check this with repairllama huggingface implementation.
            # print("repairllama_output: ", repairllama_output, repairllama_output.shape)
            # print("next repairllama token1: ", next_repairllama_token)

            next_repairllama_token = torch.where(
                input_repairllama_text_mask[:, cur_pos], repairllama_tokens[:, cur_pos], next_repairllama_token
            )

            repairllama_tokens[:, cur_pos] = next_repairllama_token

            # codellama_cur_pos = cur_pos-repairllama_start_pos-codellama_iter_start_pos

            if cur_pos - repairllama_start_pos > codellama_iter_start_pos:
                # Then the codellama logits are available.
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
                codellama_cur_pos+=1

            prev_pos = cur_pos


        repairllama_decoded = []
        for i, t in enumerate(repairllama_tokens.tolist()):

            # cut to max gen len
            t = t[len(repairllama_input_ids[i]): len(repairllama_input_ids[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.repairllama_tokenizer.eos_token_id)]
            except ValueError:
                pass
            repairllama_decoded.append(self.repairllama_tokenizer.decode(t))
        
        codellama_decoded = []
        for i, t in enumerate(codellama_tokens.tolist()):

            # cut to max gen len
            t = t[len(codellama_input_ids[i]): len(codellama_input_ids[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.codellama_tokenizer.eos_id)]
            except ValueError:
                pass
            codellama_decoded.append(self.codellama_tokenizer.decode(t))

        return repairllama_decoded, codellama_decoded
