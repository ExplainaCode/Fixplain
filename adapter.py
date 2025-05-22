import torch
import torch.nn as nn
import os
import time
import json
from pathlib import Path
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import inspect
import warnings

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
from adapter_utils import sample_top_p

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

        self.repairllama, self.repairllama_tokenizer = self._load_repairllama(
            repairllama_model_dir, repairllama_lora_dir,
            register_Attention_hooks=True)
        # print("repairllama is loaded... llama is about to load....")

        self.llama, self.llama_tokenizer = self._load_llama(
            llama_ckpt_dir, llama_max_seq_len,
            max_batch_size, llama_tokenizer,
            w_lora, lora_rank)
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.llama_tokenizer.pad_token_id)
        self.phase = phase
        self.set_trainale_params(self.phase)

        self.test_var = 0
        self.repairllama_max_seq_len = repairllama_max_seq_len
        self.llama_max_seq_len = llama_max_seq_len

    def _load_llama(
        self,
        ckpt_dir: str,
        max_seq_len: int,
        max_batch_size: int,
        tokenizer_dir: str,
        w_lora: bool,
        lora_rank: int,
    ):
        # 1) Sanity checks
        assert os.path.isdir(ckpt_dir), f"Checkpoint dir '{ckpt_dir}' not found."
        assert os.path.isdir(tokenizer_dir), f"Tokenizer dir '{tokenizer_dir}' not found."

        # 2) Read params.json (includes vocab_size=128256)
        with open(os.path.join(ckpt_dir, "params.json"), "r") as f:
            params = json.load(f)

        # 3) Build ModelArgs from those params (includes the full vocab_size)
        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            w_lora=w_lora,
            lora_rank=lora_rank,
            **params
        )

        start_time = time.time()

        # 4) Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # 5) Build the model with the original vocab_size (128256)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)

        # 6) Load all checkpoint shards with strict=False so the full embedding loads
        ckpt_paths = sorted(Path(ckpt_dir).glob("*.pth"))
        for ckpt_path in ckpt_paths:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print(f"[load] missing keys: {missing}")
            print(f"[load] unexpected keys: {unexpected}")

        # 7) Now *shrink* the embedding/output to the full tokenizer vocab
        #    (includes special tokens like EOS)
        desired_size = len(tokenizer.get_vocab())
        print("desired size_____________: ", desired_size)

        old_embed = model.tok_embeddings.weight.data
        old_output = model.output.weight.data

        # create new embedding/output matrices
        print("old embed_dim: ", old_embed.size())
        print("old_output dim: ", old_output.size())
        embed_dim = old_embed.size(1)
        new_embed = old_embed[:desired_size, :].clone()
        new_output = old_output[:desired_size, :].clone()
        print("new_embed dim: ", new_embed.size())
        print("new_output dim :", new_output.size())
        # replace modules
        model.tok_embeddings = torch.nn.Embedding(desired_size, embed_dim)
        model.tok_embeddings.weight.data.copy_(new_embed)

        model.output = torch.nn.Linear(embed_dim, desired_size, bias=False)
        model.output.weight.data.copy_(new_output)

        model.vocab_size = desired_size
        elapsed = time.time() - start_time
        print(f"Loaded and resized CodeLlama in {elapsed:.2f}s  (from 128256→{desired_size})")

        return model, tokenizer
    # def _load_llama(
    #         self, llama_ckpt_dir, 
    #         max_seq_len, max_batch_size, 
    #         llama_tokenizer, 
    #         w_lora, lora_rank
    #     ):
    #     assert os.path.isdir(llama_ckpt_dir), f"Checkpoint directory '{llama_ckpt_dir}' does not exist."
    #     assert os.path.isdir(llama_tokenizer), f"Tokenizer file '{llama_tokenizer}' does not exist."

    #     with open(os.path.join(llama_ckpt_dir, "params.json"), 'r') as f:
    #         params = json.loads(f.read())
        
    #     model_args: ModelArgs = ModelArgs(
    #         max_seq_len=max_seq_len, 
    #         max_batch_size=max_batch_size, 
    #         w_lora=w_lora, 
    #         lora_rank=lora_rank,
    #         **params
    #     )
    #     start_time = time.time()
    #     # tokenizer = Tokenizer(model_path=llama_tokenizer)
    #     tokenizer = AutoTokenizer.from_pretrained(llama_tokenizer)
    #     true_vocab_size = len(tokenizer.get_vocab())  
    #     model_args.vocab_size = true_vocab_size
    #     print("__________________________")
    #     print(model_args.vocab_size)
    #     print(true_vocab_size)
    #     print("__________________________")
    #     assert model_args.vocab_size == true_vocab_size
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    #     tokenizer.pad_token = tokenizer.eos_token
    #     # model_args.vocab_size = tokenizer.vocab_size
        
    #     torch.set_default_tensor_type(torch.cuda.HalfTensor)
    #     llama = Transformer(model_args)

    #     ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
    #     for ckpt_path in ckpts:
    #         ckpt = torch.load(ckpt_path, map_location="cpu")
    #         missing_keys, unexpected_keys = llama.load_state_dict(ckpt, strict=False)

    #     print("missing keys: ", missing_keys)
    #     print("__________________________________________-")
    #     print("unexpected_keys: ", unexpected_keys)
    #     print(f"Loaded in {time.time() - start_time:.2f} seconds")
    #     return llama, tokenizer


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
        print("______________in loading saved model params_________________")
        print(missing_keys)
        print(unexpected_keys)

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
            target_keywords = ["lora", "gate"]
            for name, para in self.llama.named_parameters():
                if any(keyword in name for keyword in target_keywords):
                    # para.data = para.data.float()
                    para.requires_grad = True
                    # print(f"Parameter: {name}, dtype: {para.dtype}")
        
        elif phase == 'inference':
            pass

        else:
            raise ValueError(f"Unknown model phase: {phase}")


    # def forward(self, repairllama_input_ids, llama_input_ids, llama_labels):
    #     # torch.autograd.set_detect_anomaly(True)

    #     repairllama_input_ids=repairllama_input_ids.to(device)
    #     llama_input_ids=llama_input_ids.to(device)
    #     llama_labels = llama_labels.to(device)

    #     _bsz, repairllama_seqlen = repairllama_input_ids.shape

    #     repairllama_h = self.repairllama.model.model.embed_tokens(repairllama_input_ids)
    #     repairllama_position_ids = torch.arange(repairllama_seqlen, dtype=torch.long, device=repairllama_input_ids.device).unsqueeze(0).expand(_bsz, -1)
    #     repairllama_mask = None
    #     repairllama_mask = torch.full((1, 1, repairllama_seqlen, repairllama_seqlen), float("-inf"), device=repairllama_h.device)
    #     repairllama_mask = torch.triu(repairllama_mask, diagonal=0 + 1).type_as(repairllama_h)
    #     # print("repairllama_mask:", repairllama_mask.dtype)

    #     # llama configuration before forward pass # This is redundent if works movw to a function or something...
    #     _bsz, llama_seqlen = llama_input_ids.shape
    #     llama_h = self.llama.tok_embeddings(llama_input_ids)
    #     llama_freq_cis = self.llama.freqs_cis.to(llama_h.device)

    #     llama_freq_cis = llama_freq_cis[:llama_seqlen]
    #     llama_casual_mask = None
    #     llama_casual_mask = torch.full((1, 1, llama_seqlen, llama_seqlen), float("-inf"), device=llama_h.device)
    #     llama_casual_mask = torch.triu(llama_casual_mask, diagonal=0 + 1).type_as(repairllama_h)

    #     assert self.repairllama.config.num_hidden_layers==self.llama.config['num_hidden_layers']
    #     n_layers = self.repairllama.config.num_hidden_layers

    #     for i in range(n_layers):
    #         repairllama_h, *_ = self.repairllama.model.model.layers[i](
    #                                             repairllama_h.contiguous(), repairllama_mask.contiguous(), repairllama_position_ids.contiguous()
    #                                         )  # Do not pass as keyword arguments since hooks don't capture inputs.   
    #         assert(self.attention_hooks_data.get(i)!=None)
    #         # with torch.no_grad():
    #         dynamic_adapter = self.attention_hooks_data[i].get('input').detach()
    #         dynamic_adapter = dynamic_adapter.to(dtype=llama_h.dtype)
    #         if torch.isnan(dynamic_adapter).any() or torch.isinf(dynamic_adapter).any():
    #             warnings.warn("dynamic adapter contains NaN or inf values.___________0", i)

    #         self.attention_hooks_data[i] = None
    #         llama_h = self.llama.layers[i](llama_h, 0, llama_freq_cis, llama_casual_mask, dynamic_adapter)
    #         if torch.isnan(llama_h).any() or torch.isinf(llama_h).any():
    #             warnings.warn("llama_h contains NaN or inf values.___________0", i)

    #     # Processing LLama output
    #     llama_h = self.llama.norm(llama_h)
    #     llama_output = self.llama.output(llama_h)
    #     llama_output = llama_output[:, :-1, :]
    #     llama_labels = llama_labels[:, 1:]

    #     if llama_labels.sum()==0 :
    #         print("llama labels sum is 0")
    #         llama_c_loss = llama_output.mean() * 0
    #     else:
    #         assert self.llama.vocab_size == self.llama_tokenizer.n_words #Do we need this line?, in load llama this is set
    #         llama_c_loss = self.criterion(llama_output.reshape(-1, self.llama.vocab_size), llama_labels.flatten())

    # def forward(self, repairllama_input_ids, repairllama_mask, llama_input_ids, llama_labels, llama_mask):
    #     # Handle device placement
    #     repairllama_input_ids = repairllama_input_ids.to(device)
    #     llama_input_ids = llama_input_ids.to(device)
    #     llama_labels = llama_labels.to(device)
    #     llama_mask = llama_mask.to(device)
    #     repairllama_mask = repairllama_mask.to(device)

    #     _bsz, repairllama_seqlen = repairllama_input_ids.shape

    #     # RepairLLama Embeddings
    #     repairllama_h = self.repairllama.model.model.embed_tokens(repairllama_input_ids)
    #     repairllama_position_ids = torch.arange(repairllama_seqlen, dtype=torch.long, device=repairllama_input_ids.device).unsqueeze(0).expand(_bsz, -1)

    #     # RepairLLama Attention Mask
    #     repairllama_attn_mask=None
    #     repairllama_attn_mask = torch.full((1, 1, repairllama_seqlen, repairllama_seqlen), float('-inf'), device=repairllama_h.device)
    #     repairllama_attn_mask = torch.triu(repairllama_attn_mask, diagonal=1)
    #     if repairllama_mask is not None:
    #         repairllama_attn_mask = repairllama_attn_mask + (repairllama_mask[:, None, None, :]).to(dtype=repairllama_h.dtype)

    #     _bsz, llama_seqlen = llama_input_ids.shape

    #     # LLaMA Embeddings
    #     llama_h = self.llama.tok_embeddings(llama_input_ids)
    #     llama_freq_cis = self.llama.freqs_cis.to(llama_h.device)[:llama_seqlen]

    #     # LLaMA Attention Mask
    #     llama_attn_mask=None
    #     llama_attn_mask = torch.full((1, 1, llama_seqlen, llama_seqlen), float('-inf'), device=llama_h.device)
    #     llama_attn_mask = torch.triu(llama_attn_mask, diagonal=1)
    #     if llama_mask is not None:
    #         llama_attn_mask = llama_attn_mask + (llama_mask[:, None, None, :]).to(dtype=llama_h.dtype)

    #     n_layers = self.repairllama.config.num_hidden_layers
    #     for i in range(n_layers):
    #         repairllama_h, *_ = self.repairllama.model.model.layers[i](
    #             repairllama_h.contiguous(), repairllama_attn_mask.contiguous(), repairllama_position_ids.contiguous()
    #         )
    #         dynamic_adapter = self.attention_hooks_data[i]['input'].detach().to(dtype=llama_h.dtype)
    #         if torch.isnan(dynamic_adapter).any() or torch.isinf(dynamic_adapter).any():
    #             warnings.warn(f'dynamic adapter contains NaN or inf values at layer {i}')

    #         llama_h = self.llama.layers[i](llama_h, 0, llama_freq_cis, llama_attn_mask, dynamic_adapter)
    #         if torch.isnan(llama_h).any() or torch.isinf(llama_h).any():
    #             warnings.warn(f'llama_h contains NaN or inf values at layer {i}')

    #     llama_h = self.llama.norm(llama_h)
    #     llama_output = self.llama.output(llama_h)[:, :-1, :]
    #     llama_labels = llama_labels[:, 1:]

    #     if llama_labels.sum() == 0:
    #         llama_c_loss = llama_output.mean() * 0
    #     else:
    #         llama_c_loss = self.criterion(llama_output.reshape(-1, self.llama.vocab_size), llama_labels.flatten())
    def forward(
        self,
        repairllama_input_ids,
        repairllama_mask,
        llama_input_ids,
        llama_labels,
        llama_mask
    ):
        # --- device placement ---
        repairllama_input_ids = repairllama_input_ids.to(device)
        llama_input_ids         = llama_input_ids.to(device)
        llama_labels            = llama_labels.to(device)
        llama_mask              = llama_mask.to(device)
        repairllama_mask        = repairllama_mask.to(device)

        bsz, repair_seqlen = repairllama_input_ids.shape
        _, llama_seqlen   = llama_input_ids.shape

        # --- RepairLLama side embeddings & masks ---
        repair_h = self.repairllama.model.model.embed_tokens(repairllama_input_ids)
        repair_pos_ids = (
            torch.arange(repair_seqlen, device=device)
                .unsqueeze(0)
                .expand(bsz, -1)
        )
        # causal mask + attention mask
        attn_inf = torch.full((1, 1, repair_seqlen, repair_seqlen),
                            float("-inf"), device=device)
        repair_attn_mask = torch.triu(attn_inf, diagonal=1)
        repair_attn_mask = repair_attn_mask + (repairllama_mask[:, None, None, :]).to(repair_h.dtype)

        # --- LLaMA side embeddings & masks ---
        llama_h      = self.llama.tok_embeddings(llama_input_ids)
        llama_freq_cis = self.llama.freqs_cis.to(device)[:llama_seqlen]
        # llama_attn_mask = torch.triu(attn_inf.expand(1,1,llama_seqlen,llama_seqlen), diagonal=1)

        attn_inf_llama = torch.full(
            (1, 1, llama_seqlen, llama_seqlen),
            float("-inf"),
            device=device
        )
        llama_attn_mask = torch.triu(attn_inf_llama, diagonal=1)
        llama_attn_mask = llama_attn_mask + (llama_mask[:, None, None, :]).to(llama_h.dtype)

        # --- pass through layers with dynamic adapter data ---
        for i in range(self.repairllama.config.num_hidden_layers):
            repair_h, *_ = self.repairllama.model.model.layers[i](
                repair_h.contiguous(),
                repair_attn_mask.contiguous(),
                repair_pos_ids.contiguous()
            )

            # pull the adapter signals you stored earlier
            dynamic_adapter = self.attention_hooks_data[i]['input'].to(dtype=llama_h.dtype)
            llama_h = self.llama.layers[i](
                llama_h, 
                0, 
                llama_freq_cis, 
                llama_attn_mask, 
                dynamic_adapter
            )

        # --- final norm + projection ---
        llama_h     = self.llama.norm(llama_h)
        logits      = self.llama.output(llama_h)  # [bsz, seqlen, vocab_size]

        # shift so tokens predict the *next* token
        logits      = logits[:, :-1, :].contiguous()        # [bsz, seqlen-1, V]
        labels      = llama_labels[:, 1:].contiguous()      # [bsz, seqlen-1]

        # --- sanity check on labels ---
        min_label, max_label = labels.min().item(), labels.max().item()
        assert min_label >= -100 and max_label < logits.size(-1), (
            f"Labels out of range: [{min_label}..{max_label}] vs vocab_size={logits.size(-1)}"
        )

        # ids    = llama_input_ids[0].tolist()          # raw input IDs including prompt+response+EOS
        # labs   = labels[0].tolist()                   # mask: -100 for prompt, real IDs for response

        # print("--- Example token-by-token dump (idx: id → label) ---")
        # for idx, (tok_id, lbl) in enumerate(zip(ids, labs)):
        #     tag = "train" if lbl != -100 else "prompt"
        #     print(f"{idx:03d}: {tok_id:5d} → {lbl:5d}   [{tag}]")
        # print("────────────────────────────────────────────────────")

        # --- compute loss with ignore_index=-100 ---
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss     = loss_fct(
            logits.view(-1, logits.size(-1)),   # [bsz*(seqlen-1), V]
            labels.view(-1)                      # [bsz*(seqlen-1)]
        )
        # ______________________________Testing____________________________
        # initialize on first call
        if not hasattr(self, "test_var"):
            self.test_var = 0
        if not hasattr(self, "write_header"):
            self.write_header = True

        # only every 100 steps
        if self.test_var % 100 == 0:
            # move logits back to CPU and pick the top tokens
            top_ids = logits[0].argmax(dim=-1).cpu().tolist()  # [seqlen-1]
            # decode input_ids and top_ids on CPU
            input_text   = self.llama_tokenizer.decode(
                llama_input_ids[0].cpu().tolist(),
                skip_special_tokens=True
            )
            generated_t  = self.llama_tokenizer.decode(
                top_ids,
                skip_special_tokens=True
            )
            # print("codellama labels (input):", input_text)
            # print("codellama decoded (output):", generated_t)
            # print(
            #     f"Gate max: {self.llama.layers[0].attention.gate.max().item():.6f}, "
            #     f"min: {self.llama.layers[0].attention.gate.min().item():.6f}"
            # )

            # append to CSV
            csv_file = "llama_results.csv"
            with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
                import csv
                writer = csv.writer(file)
                if self.write_header:
                    writer.writerow(["Step", "Input Text", "Generated Text"])
                    self.write_header = False
                writer.writerow([self.test_var, input_text, generated_t])
                print(f"Wrote record for step {self.test_var}")

        self.test_var += 1
        # _____________________________Testing____________________________

        return loss
    

    @torch.inference_mode()
    def forward_inference(self,
        llama_input_ids,        # tensor (batch=1, seq_len)
        llama_mask,        # tensor (batch=1, seq_len)
        llama_start_pos: int = 0
    ):
        llama_input_ids = llama_input_ids.to(device)
        llama_mask = llama_mask.to(device)

        # Shapes
        _bsz, llama_seqlen = llama_input_ids.shape

        # Embeddings + rotary freqs
        llama_h = self.llama.tok_embeddings(llama_input_ids)
        llama_freq_cis = self.llama.freqs_cis.to(device)[llama_start_pos: llama_start_pos + llama_seqlen]

        # 1) Build causal mask for inference
        #    shape (1, 1, seq_len, seq_len), -inf where j > i
        causal = torch.full(
            (1, 1, llama_seqlen, llama_seqlen),
            float("-inf"),
            device=device
        )
        attn_mask = torch.triu(causal, diagonal=1).type_as(llama_h)
        attn_mask = attn_mask + (llama_mask[:, None, None, :]).to(llama_h.dtype)

        # 3) Run through decoder layers
        n_layers = self.repairllama.config.num_hidden_layers
        for i in range(n_layers):
            dynamic_adapter = self.attention_hooks_data[i]['input']
            # Each llama layer expects: (hidden, start_pos, freqs, attn_mask, adapter)
            llama_h = self.llama.layers[i](
                llama_h,
                llama_start_pos,
                llama_freq_cis,
                attn_mask,
                dynamic_adapter
            )

        # 4) Final projection & decode
        llama_h = self.llama.norm(llama_h)
        llama_output = self.llama.output(llama_h).float()     # (1, seq_len, vocab)
        return llama_output

    @torch.inference_mode()
    def forward_repairllama(self,
        repairllama_input_ids,    # Tensor: (batch, seq_len)
        repairllama_mask
    ):
        bsz, seq_len = repairllama_input_ids.shape

        # Move to device
        repairllama_input_ids = repairllama_input_ids.to(device)
        repairllama_mask = repairllama_mask.to(device)

        # 2) Embeddings + positional IDs
        h = self.repairllama.model.model.embed_tokens(repairllama_input_ids)
        position_ids = (torch.arange(seq_len, device=device)
                        .unsqueeze(0)
                        .expand(bsz, -1)
        )

        # 3) Build causal mask: shape (1,1,seq,seq)
        causal = torch.full((1, 1, seq_len, seq_len),
                            float("-inf"), device=device)
        attn_mask = torch.triu(causal, diagonal=1)
        attn_mask = attn_mask + (repairllama_mask[:, None, None, :]).to(h.dtype)

        for i in range(self.repairllama.config.num_hidden_layers):
            h, *_ = self.repairllama.model.model.layers[i](
                h, attn_mask, position_ids
            )

    @torch.inference_mode()
    def generate(
        self,
        repairllama_input_ids,  # [B, L]
        repairllama_mask,       # [B, L]
        llama_input_ids,        # [B, L]
        llama_mask,             # [B, L]   ← True where INPUT was padded
        temperature: float = 0.1,
        top_p: float = 0.75,
    ):
        """
        Autoregressive generation that:
        - left-pads inputs (llama_mask marks pads)
        - samples until EOS or max length 
        - stops sampling on a per-sequence basis
        """
        repairllama_input_ids = repairllama_input_ids.to(device)
        repairllama_mask  = repairllama_mask.to(device)
        llama_input_ids = llama_input_ids.to(device)
        llama_mask = llama_mask.to(device)

        bsz, seq_len = llama_input_ids.shape
        eos_id = self.llama_tokenizer.eos_token_id

        # 1) run repair model once
        with torch.amp.autocast("cuda"):
            self.forward_repairllama(repairllama_input_ids, repairllama_mask)

        # 2) prepare finished flags
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)

        # determine earliest position we need to start decoding from
        # i.e. the first non-padded token in each row
        # we take the minimum across the batch so we can run them in lock-step
        min_prompt_start = min(
            (llama_mask[i].tolist().index(False) for i in range(bsz)),
            default=0
        )

        prev_pos = min_prompt_start

        # 3) loop token by token
        for cur_pos in range(min_prompt_start, self.llama_max_seq_len):
            segment = llama_input_ids[:, prev_pos:cur_pos]
            if segment.size(1) == 0:
                continue 
            # run only the *new* token positions
            with torch.amp.autocast("cuda"):
                logits = self.forward_inference(
                    llama_input_ids[:, prev_pos:cur_pos], 
                    llama_mask[:, prev_pos:cur_pos],
                    prev_pos
                )  # [B, seq_segment, V]

            # sample next token
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_tok = sample_top_p(probs, top_p)  # [B]
            else:
                next_tok = torch.argmax(logits[:, -1], dim=-1)  # [B]

            # ensure shape
            next_tok = next_tok.to(device)
            next_tok = next_tok.view(bsz)
            # 4) enforce pad‐positions and finished‐EOS
            pad_vals = llama_input_ids[:, cur_pos]
            next_tok = torch.where(llama_mask[:, cur_pos].to(torch.bool), pad_vals, next_tok)
            next_tok = torch.where(finished, eos_id, next_tok)

            # 5) write back and update finished
            llama_input_ids[:, cur_pos] = next_tok
            finished |= next_tok.eq(eos_id)

            # 6) break early if done
            if finished.all():
                break

            prev_pos = cur_pos

        # 4) free hook data
        self.attention_hooks_data = {}
        
        seq_ids = llama_input_ids[0].tolist()
        print("Final token IDs: ", seq_ids)

        # Raw decode (includes special tokens)
        raw_text = self.llama_tokenizer.decode(seq_ids, skip_special_tokens=False)
        print("Raw decode   : ", raw_text)

        # 5) decode each sequence up to its first EOS
        llama_decoded = []
        for seq in llama_input_ids.tolist():
            if eos_id in seq:
                seq = seq[: seq.index(eos_id)]
            else:
                # no EOS found, use full sequence
                pass
            llama_decoded.append(self.llama_tokenizer.decode(seq))

        return repairllama_input_ids, llama_decoded


        # @torch.inference_mode()
        # def generate(self, 
        #     repairllama_input_ids, repairllama_mask, 
        #     llama_input_ids, llama_mask, 
        #     temperature: float=0.1, top_p:  float=0.75
        # ):
        #     bsz = len(repairllama_input_ids)
        #     assert len(repairllama_input_ids)==len(llama_input_ids) #batch sizes should be equal.
        
        #     params = self.llama.params
        #     assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        #     min_llama_prompt_size = min([len(t[0]) for t in llama_input_ids])
        #     prev_pos = 0
        #     with torch.amp.autocast("cuda"):
        #         self.forward_repairllama(repairllama_input_ids, repairllama_mask)

        #     for cur_pos in range(min_llama_prompt_size, self.llama_max_seq_len):  
        #         with torch.amp.autocast("cuda"):
        #             llama_logits = self.forward_inference(llama_input_ids[:, prev_pos:cur_pos], prev_pos)
        #         if temperature > 0:
        #             probs = torch.softmax(llama_logits[:, -1] / temperature, dim=-1)
        #             next_llama_token = sample_top_p(probs, top_p)
        #         else:
        #             next_llama_token = torch.argmax(llama_logits[:, -1], dim=-1)
        #         next_llama_token = next_llama_token.reshape(-1)
        #         next_llama_token = torch.where(
        #             llama_mask[:, cur_pos], llama_input_ids[:, cur_pos], next_llama_token
        #         )

        #         llama_input_ids[:, cur_pos] = next_llama_token
                    
        #     self.attention_hooks_data ={} # free the memory
        #     llama_decoded = []
        #     for i, t in enumerate(llama_input_ids.tolist()):
        #         try:
        #             t = t[: t.index(self.llama_tokenizer.eos_id)]
        #         except ValueError:
        #             print("No EOS token detected!")
        #         llama_decoded.append(self.llama_tokenizer.decode(t))

        #     return repairllama_input_ids, llama_decoded
