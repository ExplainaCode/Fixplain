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

        # self.repairllama, self.repairllama_tokenizer = self._load_repairllama(
        #     repairllama_model_dir, repairllama_lora_dir,
        #     register_Attention_hooks=True)

        self.llama, self.llama_tokenizer = self._load_llama(
            llama_ckpt_dir, llama_max_seq_len,
            max_batch_size, llama_tokenizer,
            w_lora, lora_rank)
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.llama_tokenizer.pad_token_id)
        self.phase = phase
        self.set_trainale_params(self.phase)

        self.test_var = 0
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

    def forward(
        self,
        llama_input_ids,
        llama_labels,
        llama_mask
    ):
        # --- device placement ---
        llama_input_ids         = llama_input_ids.to(device)
        llama_labels            = llama_labels.to(device)
        llama_mask              = llama_mask.to(device)

        _, llama_seqlen   = llama_input_ids.shape

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
        for i in range(self.llama.config.num_hidden_layers): #32
            # pull the adapter signals you stored earlier
            dynamic_adapter = None
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

    # @torch.inference_mode()
    # def generate(
    #     self,
    #     repairllama_input_ids,  # [B, L]
    #     repairllama_mask,       # [B, L]
    #     llama_input_ids,        # [B, L]
    #     llama_mask,             # [B, L]   ← True where INPUT was padded
    #     max_gen_len: int =128,
    #     temperature: float = 0.1,
    #     top_p: float = 0.75,
    # ):
    #     """
    #     Autoregressive generation that:
    #     - left-pads inputs (llama_mask marks pads)
    #     - samples until EOS or max length 
    #     - stops sampling on a per-sequence basis
    #     """
    #     repairllama_input_ids = repairllama_input_ids.to(device)
    #     repairllama_mask  = repairllama_mask.to(device)
    #     llama_input_ids = llama_input_ids.to(device)
    #     llama_mask = llama_mask.to(device)

    #     bsz, seq_len = llama_input_ids.shape
    #     eos_id = self.llama_tokenizer.eos_token_id

    #     # 1) run repair model once
    #     with torch.amp.autocast("cuda"):
    #         self.forward_repairllama(repairllama_input_ids, repairllama_mask)

    #     prev_pos = 0

    #     # 3) loop token by 
    #     for cur_pos in range(self.llama_max_seq_len, self.llama_max_seq_len+max_gen_len):
    #         with torch.amp.autocast("cuda"):
    #             logits = self.forward_inference(
    #                 llama_input_ids[:, prev_pos:cur_pos], 
    #                 llama_mask[:, prev_pos:cur_pos],
    #                 prev_pos
    #             )  # [B, seq_segment, V]

    #         # sample next token
    #         if temperature > 0:
    #             probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
    #             next_tok = sample_top_p(probs, top_p)  # [B]
    #         else:
    #             next_tok = torch.argmax(logits[:, -1], dim=-1)  # [B]

    #         # ensure shape
    #         next_tok = next_tok.to(device)
    #         next_tok = next_tok.view(bsz)

    #         next_tok_col = next_tok.unsqueeze(1)
    #         llama_input_ids = torch.cat([llama_input_ids, next_tok_col], dim=1)

    #         new_mask_col = torch.zeros((bsz, 1), dtype=torch.bool, device=device)
    #         llama_mask    = torch.cat([llama_mask, new_mask_col], dim=1)

    #         prev_pos = cur_pos

    #     # 4) free hook data
    #     self.attention_hooks_data = {}

    #     seq_ids = llama_input_ids[0].tolist()
    #     print("Final token IDs: ", seq_ids)

    #     # Raw decode (includes special tokens)
    #     raw_text = self.llama_tokenizer.decode(seq_ids, skip_special_tokens=False)
    #     print("Raw decode   : ", raw_text)

    #     # 5) decode each sequence up to its first EOS
    #     llama_decoded = []
    #     for seq in llama_input_ids.tolist():
    #         if eos_id in seq:
    #             seq = seq[: seq.index(eos_id)]
    #         else:
    #             # no EOS found, use full sequence
    #             pass
    #         llama_decoded.append(self.llama_tokenizer.decode(seq))

    #     return repairllama_input_ids, llama_decoded


    @torch.inference_mode()
    def generate(
        self,
        repairllama_input_ids,  # [B, L]
        repairllama_mask,       # [B, L]
        llama_input_ids,        # [B, L]
        llama_mask,             # [B, L]   ← True where INPUT was padded
        batch_size: int = 1,
        max_gen_len: int = 128,
        temperature: float = 0.1,
        top_p: float = 0.75,
    ):
        """
        Inefficient but correct autoregressive generation:
        - run the repair model once
        - then for each new token, re-run forward_inference on the entire prefix
        - sample from the last logit and append
        - stop on EOS or max length
        """
        repairllama_input_ids = repairllama_input_ids.to(device)
        repairllama_mask  = repairllama_mask.to(device)
        llama_input_ids = llama_input_ids.to(device)
        llama_mask = llama_mask.to(device)

        eos_id = self.llama_tokenizer.eos_token_id

        # 1) Run the repair model on the prompt
        with torch.amp.autocast("cuda"):
            _ = self.forward_repairllama(repairllama_input_ids, repairllama_mask)

        # 2) Autoregressive loop
        for _step in range(max_gen_len):
            # (Re)run the entire sequence through forward_inference
            with torch.amp.autocast("cuda"):
                logits = self.forward_inference(
                    llama_input_ids,
                    llama_mask,
                    llama_start_pos=0,         # always start at 0
                )                          # → [B, seq_len, V]

            # pick the distribution over the *last* position
            last_logits = logits[:, -1]    # [B, V]

            if temperature > 0:
                probs = torch.softmax(last_logits / temperature, dim=-1)
                next_tok = sample_top_p(probs, top_p)      # [B]
            else:
                next_tok = torch.argmax(last_logits, dim=-1)  # [B]

            # append new token to input_ids and mask
            next_tok = next_tok.to(device).view(batch_size, 1)   # [B,1]
            # print("ids:", llama_input_ids.shape, " new tok:", next_tok.shape)

            llama_input_ids = torch.cat([llama_input_ids, next_tok], dim=1)
            llama_mask      = torch.cat([llama_mask,
                                        torch.zeros_like(next_tok, dtype=torch.bool)],
                                        dim=1)

            # stop early if every sequence produced EOS
            if (next_tok == eos_id).all():
                break

        # 3) clear any saved cache/hooks
        self.attention_hooks_data = {}

        # seq_ids = llama_input_ids[0].tolist()
        # print("Final token IDs: ", seq_ids)

        # # Raw decode (includes special tokens)
        # raw_text = self.llama_tokenizer.decode(seq_ids, skip_special_tokens=False)
        # print("Raw decode   : ", raw_text)

        # 4) decode each sequence up to its first EOS
        outputs = []
        for seq in llama_input_ids.tolist():
            # take only tokens *after* position seq_len
            gen_portion = seq[self.llama_max_seq_len:]
            # if there’s an EOS in there, cut at EOS
            outputs.append(self.llama_tokenizer.decode(gen_portion))

        return repairllama_input_ids, outputs
