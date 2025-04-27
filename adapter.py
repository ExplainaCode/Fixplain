import torch
import torch.nn as nn
import os
import time
import json
from pathlib import Path
import warnings

from .llama.model import ModelArgs, Transformer
from .llama.tokenizer import Tokenizer
from .adapter_utils import sample_top_p
from peft import PeftModel
from transformers import (
AutoTokenizer,
AutoModelForCausalLM,
GenerationConfig,
HfArgumentParser,
BitsAndBytesConfig,
)
import inspect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import inspect

def debug_info(message:str = None):
    frame = inspect.currentframe().f_back
    print(f"Debug: File '{inspect.getfile(frame)}', Line {frame.f_lineno}")
    if message:
        print(f"Message: {message}")

class LLamaAdapter(nn.Module):
    def __init__(self,
                 llama_ckpt_dir, llama_tokenizer,
                 repairllama_lora_dir='./repairllama-lora', repairllama_model_dir="codellama/CodeLlama-7b-hf",
                 max_seq_len=512, max_batch_size=2,
                 w_bias=False,
                 w_lora=False, lora_rank=16, 
                 phase="inference",):
        super().__init__()
        self.attention_hooks_data = {}

        self.repairllama, self.repairllama_tokenizer = self._load_repairllama(
            repairllama_model_dir, repairllama_lora_dir,
            register_Attention_hooks=True)
        # print("repairllama is loaded... llama is about to load....")

        self.llama, self.llama_tokenizer = self._load_llama(
            llama_ckpt_dir, max_seq_len,
            max_batch_size, llama_tokenizer,
            w_lora, lora_rank)
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.llama_tokenizer.pad_id)
        self.phase = phase
        self.set_trainale_params(self.phase)

        self.test_var = 0

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
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        assert model_args.vocab_size == tokenizer.n_words
        tokenizer.pad_id = tokenizer.eos_id
        model_args.vocab_size = tokenizer.n_words
        
        # if torch.cuda.is_bf16_supported():
            # torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        # else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        llama = Transformer(model_args)

        # Print data type of model parameters
        # for name, param in llama.named_parameters():
        #     print(f"Parameter: {name}, dtype: {param.dtype}")
        ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
        for ckpt_path in ckpts:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            missing_keys, unexpected_keys = llama.load_state_dict(ckpt, strict=False)

            # debug_info("_"*20)
            # print("Missing Keys (not updated):", missing_keys)
            # print("Unexpected Keys (not in model):", unexpected_keys)
            # debug_info("_"*20)
                
        # for name, param in llama.state_dict().items():
        #     print(f"Parameter: {name}, dtype: {param.dtype}")
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

        # debug_info("____________________in trained weights loading___________________")
        # print(f"Checkpoint: {ckpt_path}")
        # print("Expected Keys (Model Parameters):", set(self.llama.state_dict().keys()))
        # debug_info("_______________________________________")
        # print("Missing Keys (not updated):", missing_keys)
        # debug_info("_______________________________________")
        # print("Unexpected Keys (not in model):", unexpected_keys)
        # debug_info("-" * 20)


    def _hook_fn(self, module, input, output):
        """
        Hook function to capture inputs of attention layers.
        """
        layer_id = module.layer_id
        self.attention_hooks_data[layer_id] = { # {0:{"input": (x, )}}
            "input": input[0].detach(),
        }
        # if (layer_id==0):
        #     print("__________")
        #     print(self.attention_hooks_data[0])
        #     print(self.attention_hooks_data[0].get('input').shape)
        #     exit(0)
        

    def set_trainale_params(self, phase='inference'):
        for name, para in self.named_parameters():
            para.requires_grad = False

        if phase == 'finetune':
            target_keywords = ["lora", "gate"]
            for name, para in self.llama.named_parameters():
                if any(keyword in name for keyword in target_keywords):
                    # para.data = para.data.float()
                    para.requires_grad = True

                    # debug_info("-"*20 + "Trainable parameters" + "-"*20)
                    # print(f"Parameter: {name}, dtype: {para.dtype}")
        
        elif phase == 'inference':
            pass

        else:
            raise ValueError(f"Unknown model phase: {phase}")


    def forward(self, repairllama_input_ids, llama_input_ids, llama_labels, optimizer=None):
        # torch.autograd.set_detect_anomaly(True)

        repairllama_input_ids=repairllama_input_ids.to(device)
        llama_input_ids=llama_input_ids.to(device)
        llama_labels = llama_labels.to(device)

        _bsz, repairllama_seqlen = repairllama_input_ids.shape

        repairllama_h = self.repairllama.model.model.embed_tokens(repairllama_input_ids)
        # print("repairllama_h dtype:", repairllama_h.dtype)
        # print(repairllama_h.shape)
        # print(repairllama_h)
        # repairllama_freqs_cis = self.repairllama.freqs_cis.to(repairllama_h.device) 
        # repairllama_freqs_cis = repairllama_freqs_cis[:repairllama_seqlen]
        repairllama_position_ids = torch.arange(repairllama_seqlen, dtype=torch.long, device=repairllama_input_ids.device).unsqueeze(0).expand(_bsz, -1)
        repairllama_mask = None
        repairllama_mask = torch.full((1, 1, repairllama_seqlen, repairllama_seqlen), float("-inf"), device=repairllama_h.device)
        repairllama_mask = torch.triu(repairllama_mask, diagonal=0 + 1).type_as(repairllama_h)
        # print("repairllama_mask:", repairllama_mask.dtype)


        # llama configuration before forward pass # This is redundent if works movw to a function or something...
        _bsz, llama_seqlen = llama_input_ids.shape
        # debug_info(llama_input_ids.shape)
        # print(llama_input_ids)
        llama_h = self.llama.tok_embeddings(llama_input_ids)
        # print("llama_h dtype:", llama_h.dtype)

        # debug_info("llama h")
        # print(llama_h)
        llama_freq_cis = self.llama.freqs_cis.to(llama_h.device)

        llama_freq_cis = llama_freq_cis[:llama_seqlen]
        llama_mask = None
        llama_mask = torch.full((1, 1, llama_seqlen, llama_seqlen), float("-inf"), device=llama_h.device)
        llama_mask = torch.triu(llama_mask, diagonal=0 + 1).type_as(repairllama_h)
        # print("llama_freq_cis dtype:", llama_freq_cis.dtype)
        # print("llama_mask dtype:", llama_mask.dtype)
        # print(llama_mask)

        assert self.repairllama.config.num_hidden_layers==self.llama.config['num_hidden_layers']
        n_layers = self.repairllama.config.num_hidden_layers

        for i in range(n_layers):
            repairllama_h, *_ = self.repairllama.model.model.layers[i](
                                                repairllama_h.contiguous(), repairllama_mask.contiguous(), repairllama_position_ids.contiguous()
                                            )  # Do not pass as keyword arguments since hooks don't capture inputs.   
            assert(self.attention_hooks_data.get(i)!=None)
            # with torch.no_grad():
            dynamic_adapter = self.attention_hooks_data[i].get('input').detach()
            dynamic_adapter = dynamic_adapter.to(dtype=llama_h.dtype)
            if torch.isnan(dynamic_adapter).any() or torch.isinf(dynamic_adapter).any():
                warnings.warn("dynamic adapter contains NaN or inf values.___________0", i)
            # del self.attention_hooks_data[i]
            self.attention_hooks_data[i] = None
            llama_h = self.llama.layers[i](llama_h, 0, llama_freq_cis, llama_mask, dynamic_adapter)
            if torch.isnan(llama_h).any() or torch.isinf(llama_h).any():
                warnings.warn("llama_h contains NaN or inf values.___________0", i)
        # self.attention_hooks_data={}
        # print("second repairllama_h dtype:", repairllama_h.dtype)

        # Processing RepairLLama output
        # repairllama_h = self.repairllama.model.model.norm(repairllama_h) # Why do even need this line?
        # repairllama_output = self.repairllama.model.lm_head(repairllama_h[:, -1, :]) # Why do even need this line?
        # repairllama_output = repairllama_output[:, :-1, :]
        # repairllama_labels = repairllama_labels[:, 1:]

        # if repairllama_labels.sum() == 0:
        #     reapirllama_c_loss = repairllama_output.mean() * 0
        # else:
        #     assert self.repairllama.vocab_size == 32000
        #     reapirllama_c_loss = self.criterion(repairllama_output.reshape(-1, self.repairllama.vocab_size), repairllama_labels.flatten())

        # Processing LLama output

        llama_h = self.llama.norm(llama_h)
        # debug_info("after normalization")
        # print(llama_h)
        llama_output = self.llama.output(llama_h)
        # debug_info("after output layer")
        # print(llama_output.float())
    
        # next_llama_token = torch.argmax(llama_output[:, 0:1, :], dim=-1)
        # print(next_llama_token)
        llama_output = llama_output[:, :-1, :]
        llama_labels = llama_labels[:, 1:]

        if llama_labels.sum()==0 :
            print("llama labels sum is 0")
            llama_c_loss = llama_output.mean() * 0
        else:
            assert self.llama.vocab_size == self.llama_tokenizer.n_words #Do we need this line?, in load llama this is set
            llama_c_loss = self.criterion(llama_output.reshape(-1, self.llama.vocab_size), llama_labels.flatten())
        # print("llama_output shape:", llama_output.shape)
        # print("llama_labels shape:", llama_labels.shape)

        # ______________________________Testing____________________________
        if self.test_var <= 1:
            # print("llama output shape: ", llama_output.shape)
            # print("llama labels shape: ", llama_labels.shape)
            # print("llama input ids: ", llama_input_ids)
            llama_input = self.llama_tokenizer.decode(llama_input_ids[0].tolist())
            # print("llama input ids (for 0 th example in the atch): ", self.llama_tokenizer.decode(llama_input_ids[0].tolist()))
            # print("llama_output (for 0 th output): ",  llama_output[0])
            token_ids = llama_output[0].argmax(dim=-1).tolist()  # Get token IDs
            decoded_text = self.llama_tokenizer.decode(token_ids) 
            # llama_decoded = []
            # for i, t in enumerate(llama_output[0].tolist()):
            #     # cut to max gen len
            #     # t = t[len(llama_input_ids[i]): len(llama_input_ids[i]) + max_gen_len]
            #     # cut to eos tok if any
            #     try:
            #         t = t[: t.index(self.llama_tokenizer.eos_id)]
            #     except ValueError:
            #         pass
            #     llama_decoded.append(self.llama_tokenizer.decode(t))

            # print("llama_decoded: " , llama_decoded)
            print ("llama decoded: ", decoded_text)
            csv_file = "llama_results.csv"
            write_header = not os.path.exists(csv_file)
            
            with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
                import csv
                writer = csv.writer(file)
                
                # Write the header only on the first iteration
                if write_header:
                    writer.writerow(["Input Text", "Generated Text"])
                    write_header = False  # Ensure header is not written again

                # Write the data for this iteration
                writer.writerow([llama_input, decoded_text])
                print(f"Wrote record: {self.test_var}")
            self.test_var+=1
        # _____________________________Testing____________________________

        return llama_c_loss
    
    @torch.inference_mode()
    def forward_inference(self, llama_input_ids,llama_start_pos:int):
        llama_input_ids=llama_input_ids.to(device)

        _bsz, llama_seqlen = llama_input_ids.shape
        # debug_info(llama_input_ids.shape)
        # print(llama_input_ids)
        llama_h = self.llama.tok_embeddings(llama_input_ids)
        # debug_info(llama_h.shape)
        # print(llama_h)
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
            # if n_layers==31:
            #     debug_info(f"{i}")
            #     print(llama_h)

        llama_h = self.llama.norm(llama_h)
        # debug_info("after norm")
        # print(llama_h)
        llama_output = self.llama.output(llama_h).float()
        # debug_info("llama output")
        # print(llama_output)
        token_ids = llama_output[0].argmax(dim=-1).tolist()  # Get token IDs
        # token_ids=[token_ids]
        decoded_text = self.llama_tokenizer.decode(token_ids)
        # debug_info(decoded_text)
        next_llama_token = torch.argmax(llama_output[:, -1], dim=-1)
        # debug_info("true decoding")
        # print(next_llama_token)
        # print(self.llama_tokenizer.decode(next_llama_token.tolist()))
        return llama_output

    @torch.inference_mode()
    def forward_repairllama(self, repairllama_input_ids):

        import torch.nn.functional as F
        seq_len = repairllama_input_ids.shape[-1]
        pad_len = 1024 - seq_len  # Calculate how much padding is needed

        if pad_len > 0:
            repairllama_input_ids = F.pad(repairllama_input_ids, (pad_len, 0))
        # debug_info("____________________________________")
        # print(repairllama_input_ids)
        # print(repairllama_input_ids.shape)
        repairllama_input_ids=repairllama_input_ids.to(device)
        _bsz, repairllama_seqlen = repairllama_input_ids[0].shape

        # debug_info(repairllama_input_ids[0].shape)
        # print(repairllama_input_ids[0])
        repairllama_h = self.repairllama.model.model.embed_tokens(repairllama_input_ids[0]) # apass through embedding layer
        # debug_info(repairllama_h.shape)
        # print(repairllama_h)

        # repairllama_freqs_cis = self.repairllama.freqs_cis.to(repairllama_h.device) 
        # repairllama_freqs_cis = repairllama_freqs_cis[:repairllama_seqlen]
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
        # debug_info("llama actual input decoded")
        # print(self.llama_tokenizer.decode(llama_input_ids))
        # llama_input_copy = llama_input_ids[:1]
        # llama_input_ids=None
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

            # if len(llama_input_copy)>cur_pos:
            #     llama_tokens[:, cur_pos] = llama_input_copy[cur_pos]
            # else:
            llama_tokens[:, cur_pos] = next_llama_token
                
            # prev_pos = cur_pos    
            # if i>3:                  #----------for deugging
            #     break # for debugging
            # i+=1
            # print("___________________________")
        self.attention_hooks_data ={} # free the memory
        # print("llama_tokens: ",  llama_tokens)
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
