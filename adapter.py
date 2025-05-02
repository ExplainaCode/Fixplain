import torch
import torch.nn as nn
import os
import time
import json
import csv
from pathlib import Path
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import inspect
import warnings

from .llama.model import ModelArgs, Transformer
from .llama.tokenizer import Tokenizer
from .adapter_utils import sample_top_p

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def debug_info(message:str = None):
    frame = inspect.currentframe().f_back
    print(f"Debug: File '{inspect.getfile(frame)}', Line {frame.f_lineno}")
    if message:
        print(f"Message: {message}")

class LLamaAdapter(nn.Module):
    def __init__(self,
        llama_ckpt_dir, llama_tokenizer,
        repairllama_lora_dir='./repairllama-lora', repairllama_model_dir="codellama/CodeLlama-7b-hf",
        max_batch_size=2,
        w_bias=False,
        w_lora=False, lora_rank=16, 
        phase="inference",
        repairllama_max_seq_len=1024,
        llama_max_seq_len=256,
    ):
        super().__init__()
        self.attention_hooks_data = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.repairllama, self.repairllama_tokenizer = self._load_repairllama(
            repairllama_model_dir, repairllama_lora_dir,
            register_Attention_hooks=True)
        # print("repairllama is loaded... llama is about to load....")

        self.llama, self.llama_tokenizer = self._load_llama(
            llama_ckpt_dir, llama_max_seq_len,
            max_batch_size, llama_tokenizer,
            w_lora, lora_rank)
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.llama_tokenizer.pad_id)
        self.phase = phase
        self.set_trainale_params(self.phase)

        self.test_var = 0
        self.repairllama_max_seq_len = repairllama_max_seq_len
        self.llama_max_seq_len = llama_max_seq_len

    def _load_llama(
            self, llama_ckpt_dir, 
            max_seq_len, max_batch_size, 
            llama_tokenizer, 
            w_lora, lora_rank
        ):
        assert os.path.isdir(llama_ckpt_dir), f"Checkpoint directory '{llama_ckpt_dir}' does not exist."
        assert os.path.isfile(llama_tokenizer), f"Tokenizer file '{llama_tokenizer}' does not exist."

        with open(os.path.join(llama_ckpt_dir, "params.json"), 'r') as f:
            params = json.loads(f.read())
        
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, 
            max_batch_size=max_batch_size, 
            w_lora=w_lora, 
            lora_rank=lora_rank,
            **params
        )
        start_time = time.time()
        tokenizer = Tokenizer(model_path=llama_tokenizer)
        assert model_args.vocab_size == tokenizer.n_words
        tokenizer.pad_id = tokenizer.eos_id
        model_args.vocab_size = tokenizer.n_words
        
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        llama = Transformer(model_args)

        ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
        for ckpt_path in ckpts:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            missing_keys, unexpected_keys = llama.load_state_dict(ckpt, strict=False)

        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        return llama, tokenizer


    def _load_repairllama(self, repairllama_model_dir, repairllama_lora_dir, register_Attention_hooks=True):
        tokenizer = AutoTokenizer.from_pretrained(repairllama_model_dir, 
                                                  trust_remote_code=True,
                                                  padding_size='left')
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

        repairllama = AutoModelForCausalLM.from_pretrained(
            repairllama_model_dir,
            torch_dtype=torch.float16,
            # load_in_8bit=True, # commented initially
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            ),
            # quantization_config=None,
            device_map="auto",
        )

        repairllama = PeftModel.from_pretrained(
            repairllama,
            repairllama_lora_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        repairllama.config.pad_token = tokenizer.pad_token = tokenizer.unk_token 

        if register_Attention_hooks:
            """
            Registers hooks on all LlamaSdpaAttention modules.
            """
            layer_id = 0
            for layer in repairllama.model.model.layers: # Hooks is registered to layer lock not to attention lock - no harm
                layer.layer_id = layer_id  # Tag the layer with an ID
                layer.register_forward_hook(self._hook_fn)
                layer_id += 1

        return repairllama, tokenizer
    
    def load_codellma_tuned(self, llama_trained_weight_dir):
        ckpts = sorted(Path(llama_trained_weight_dir).glob("*.pth"))
        ckpt_path = ckpts[-1]
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        ckpt = torch.load(ckpt_path, map_location="cpu") # This ckeckpoint contains other parameters as well
        ckpt = ckpt["model"] 
        missing_keys, unexpected_keys = self.llama.load_state_dict(ckpt, strict=False)

    def _hook_fn(self, module, input, output):
        """
        Hook function to capture inputs of attention layers.
        """
        layer_id = module.layer_id
        self.attention_hooks_data[layer_id] = { # {0:{"input": (x, )}}
            "input": input[0].detach(),
        }

    def set_trainale_params(self, phase='inference'):
        for name, para in self.named_parameters():
            para.requires_grad = False

        if phase == 'finetune':
            target_keywords = ["lora", "gate", "adapter"]
            for name, para in self.llama.named_parameters():
                if any(keyword in name for keyword in target_keywords):
                    # para.data = para.data.float()
                    para.requires_grad = True
                    print(f"Parameter: {name}, dtype: {para.dtype}")
        
        elif phase == 'inference':
            pass

        else:
            raise ValueError(f"Unknown model phase: {phase}")

    def fwd_llama(self, llama_input_ids):
        # Ensure inputs are on the correct device (assuming model is already on device)
        # repairllama_input_ids = repairllama_input_ids.to(self.device)
        llama_input_ids = llama_input_ids.to(self.device)

        # bsz, repairllama_seqlen = repairllama_input_ids.shape
        bsz, llama_seqlen = llama_input_ids.shape

        # Precompute common elements for llama
        llama_h = self.llama.tok_embeddings(llama_input_ids)
        llama_freq_cis = self.llama.freqs_cis[:llama_seqlen].to(llama_h.device)
        llama_mask = self._prepare_decoder_attention_mask(
            llama_h.shape[:2], llama_h.dtype, llama_h.device
        )

        for i in range(self.llama.config.num_hidden_layers):
            # Dynamic adapter from hooks
            dynamic_adapter = self.attention_hooks_data[i].get('input').detach()
            dynamic_adapter = dynamic_adapter.to(llama_h.dtype)
            
            # Llama layer
            llama_h = self.llama.layers[i](
                llama_h, 0, llama_freq_cis, llama_mask, dynamic_adapter
            )

        # Final processing
        llama_h = self.llama.norm(llama_h)
        llama_output = self.llama.output(llama_h)[:, :-1, :]  # Shift left

        # Optional logging (optimized)
        if self.test_var % 100 == 0:
            self._log_inference(llama_input_ids, llama_output)
        
        return llama_output
    
    def fwd_repairllama(self, repairllama_input_ids):
        repairllama_input_ids = repairllama_input_ids.to(self.device)
        bsz, repairllama_seqlen = repairllama_input_ids.shape

        repairllama_h = self.repairllama.model.model.embed_tokens(repairllama_input_ids)
        repairllama_position_ids = torch.arange(repairllama_seqlen, dtype=torch.long, device=self.device).unsqueeze(0).expand(bsz, -1)
        repairllama_mask = self._prepare_decoder_attention_mask(
            repairllama_h.shape[:2], repairllama_h.dtype, repairllama_h.device
        )

        for i in range(self.llama.config['num_hidden_layers']):
            # RepairLlama layer
            repairllama_h = self.repairllama.model.model.layers[i](
                repairllama_h, repairllama_mask, repairllama_position_ids
        )[0]

    def forward(self, repairllama_input_ids, llama_input_ids, llama_labels, 
                scheduled_sampling=False, sampling_rate=0.5):
        
        with torch.no_grad():
            self.fwd_repairllama(repairllama_input_ids=repairllama_input_ids)
        # Scheduled sampling decision
        use_predicted = scheduled_sampling and (torch.rand(1).item() < sampling_rate)
        if use_predicted:
            with torch.no_grad():
                # Initial teacher-forced prediction
                initial_output = self.fwd_llama(llama_input_ids)
                predicted_ids = initial_output.argmax(dim=-1)
                # Maintain sequence length with start token
                new_input = torch.cat([llama_input_ids[:, :1], predicted_ids], dim=1)
            
            # Forward pass with predicted inputs
            llama_output = self.fwd_llama(new_input)
        else:
            # Regular teacher-forced forward pass
            llama_output = self.fwd_llama(llama_input_ids)

        # Loss calculation with padding handling
        shifted_labels = llama_labels[:, 1:].contiguous()
        llama_c_loss = self.criterion(
            llama_output.view(-1, self.llama.vocab_size),
            shifted_labels.view(-1)
        )
        
        return llama_c_loss

    # Helper methods
    def _prepare_decoder_attention_mask(self, shape, dtype, device):
        mask = torch.full((1, 1, *shape[-2:]), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1).to(dtype)

    def _log_inference(self, input_ids, output):
        input_text = self.llama_tokenizer.decode(input_ids[0].tolist())
        pred_text = self.llama_tokenizer.decode(output[0].argmax(dim=-1).tolist())
        
        # Efficient logging (consider using a logger instead)
        with open("llama_results.csv", "a") as f:
            writer = csv.writer(f)
            if not hasattr(self, '_header_written'):
                writer.writerow(["Input", "Output"])
                self._header_written = True
            writer.writerow([input_text, pred_text])
        
        self.test_var += 1
    
    @torch.inference_mode()
    def forward_inference(self, llama_input_ids,llama_start_pos:int):
        llama_input_ids=llama_input_ids.to(device)

        _bsz, llama_seqlen = llama_input_ids.shape
        llama_h = self.llama.tok_embeddings(llama_input_ids)
        llama_freq_cis = self.llama.freqs_cis.to(llama_h.device)
        llama_freq_cis = self.llama.freqs_cis[llama_start_pos : llama_start_pos + llama_seqlen]

        llama_mask=None
        if llama_seqlen>1:
            llama_mask = torch.full((llama_seqlen, llama_seqlen), float("-inf"), device=llama_h.device)
            llama_mask = torch.triu(llama_mask, diagonal=1).type_as(llama_h)
            llama_mask = torch.hstack(
                [torch.zeros((llama_seqlen, llama_start_pos), device=llama_h.device), llama_mask]
            ).type_as(llama_h)

        n_layers = self.repairllama.config.num_hidden_layers

        for i in range(n_layers):
            dynamic_adapter  = self.attention_hooks_data[i].get('input') # Hooked input to the respective repairllama layer
            llama_h = self.llama.layers[i](llama_h, llama_start_pos, llama_freq_cis, llama_mask, dynamic_adapter)

        llama_h = self.llama.norm(llama_h)
        llama_output = self.llama.output(llama_h).float()
        token_ids = llama_output[0].argmax(dim=-1).tolist()  # Get token IDs
        decoded_text = self.llama_tokenizer.decode(token_ids)
        next_llama_token = torch.argmax(llama_output[:, -1], dim=-1)
        return llama_output

    @torch.inference_mode()
    def forward_repairllama(self, repairllama_input_ids):

        import torch.nn.functional as F
        seq_len = repairllama_input_ids.shape[-1]
        pad_len = 1024 - seq_len  # Calculate how much padding is needed

        if pad_len > 0:
            repairllama_input_ids = F.pad(repairllama_input_ids, (pad_len, 0))

        repairllama_input_ids=repairllama_input_ids.to(device)
        _bsz, repairllama_seqlen = repairllama_input_ids[0].shape

        repairllama_h = self.repairllama.model.model.embed_tokens(repairllama_input_ids[0]) # apass through embedding layer
        repairllama_position_ids = torch.arange(repairllama_seqlen, dtype=torch.long, device=repairllama_input_ids.device).unsqueeze(0).expand(_bsz, -1)
        repairllama_mask = None
        repairllama_mask = torch.full((1, 1, repairllama_seqlen, repairllama_seqlen), float("-inf"), device=repairllama_h.device)
        repairllama_mask = torch.triu(repairllama_mask, diagonal=0 + 1).type_as(repairllama_h) #this should change.
        # print(repairllama_mask)
        n_layers = self.repairllama.config.num_hidden_layers
        for i in range(n_layers):
            repairllama_h, *_ = self.repairllama.model.model.layers[i](
                                                repairllama_h.contiguous(), repairllama_mask.contiguous(), repairllama_position_ids.contiguous()
                                            )  # Do not pass as keyword arguments since hooks don't capture inputs.
    @torch.inference_mode()
    def generate(self, repairllama_input_ids, llama_input_ids=None,
                   max_gen_len: int=256, max_llama_gen_len: int=125, temperature: float=0.1,
                   top_p:  float=0.75):
        bsz = len(repairllama_input_ids)
        if llama_input_ids==None:
            llama_input_ids = [
                torch.full((1, 1), fill_value=self.llama_tokenizer.bos_id, dtype=torch.long)
                for _ in range(bsz)
            ]
        assert len(repairllama_input_ids)==len(llama_input_ids) #batch sizes should be equal.
       
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        if isinstance(repairllama_input_ids[0], str): # if the inputs are given as strings instead of input_ids
             #This assumes list of pytorch tensors returns given enumerable (list) of input texts.
            repairllama_input_ids = [self.repairllama_tokenizer.encode(x, return_tensors='pt') for x in repairllama_input_ids]
        
        if isinstance(llama_input_ids[0], str):
            # This has custom tokenizer encode in llama directory
            llama_input_ids = [self.llama_tokenizer.encode(x, bos=True, eos=False) for x in llama_input_ids]

        #Clipplig to max_seq_len
        # Convert list of tensors into a single tensor
        repairllama_input_ids = torch.stack(repairllama_input_ids)
        llama_input_ids = torch.stack(llama_input_ids)
        repairllama_input_ids = repairllama_input_ids[:, :, :params.max_seq_len]
        llama_input_ids = llama_input_ids[:, :, :params.max_seq_len]

        min_llama_prompt_size = min([len(t[0]) for t in llama_input_ids])
        max_llama_prompt_size = max([len(t[0]) for t in llama_input_ids])

        total_llama_len = min(params.max_seq_len, max_llama_gen_len + max_llama_prompt_size) # instead of generic params.max_seq_len consider using specific to llama & max_gen_len for llama text.
        llama_tokens = torch.full((bsz, total_llama_len), self.llama_tokenizer.pad_id).cuda().long()

        # Copy prompts into llama_tokens - Check this
        for i in range(bsz):
            prompt = llama_input_ids[i]
            llama_tokens[i, :len(prompt[0])] = prompt[0]
            
        input_llama_text_mask = llama_tokens != self.llama_tokenizer.pad_id
        llama_start_pos = min_llama_prompt_size

        prev_pos = 0
        with torch.cuda.amp.autocast():
            # print("repairllama input ids: ", repairllama_input_ids)
            self.forward_repairllama(repairllama_input_ids)
        # i = 0
        for cur_pos in range(llama_start_pos, total_llama_len):  
            with torch.cuda.amp.autocast():
                llama_logits = self.forward_inference(llama_tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(llama_logits[:, -1] / temperature, dim=-1)
                next_llama_token = sample_top_p(probs, top_p)
            else:
                next_llama_token = torch.argmax(llama_logits[:, -1], dim=-1)
            next_llama_token = next_llama_token.reshape(-1)
            next_llama_token = torch.where(
                input_llama_text_mask[:, cur_pos], llama_tokens[:, cur_pos], next_llama_token
            )

            llama_tokens[:, cur_pos] = next_llama_token
                
        self.attention_hooks_data ={} # free the memory
        llama_decoded = []
        for i, t in enumerate(llama_tokens.tolist()):

            # cut to max gen len
            t = t[len(llama_input_ids[i]): len(llama_input_ids[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.llama_tokenizer.eos_id)]
            except ValueError:
                pass
            llama_decoded.append(self.llama_tokenizer.decode(t))

        return repairllama_input_ids, llama_decoded