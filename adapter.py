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
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.codellama, self.codellama_tokenizer = self._load_codellama(
            codellama_ckpt_dir, max_seq_len, max_batch_size, codellama_tokenizer)
        self.repairllama, self.repairllama_tokenizer = self._load_repairllama(
            repairllama_model_dir, repairllama_lora_dir,
            register_attention_hooks=True)

    def _load_codellama(self, codellama_ckpt_dir, max_seq_len, max_batch_size, codellama_tokenizer):
        with open(os.path.join(codellama_ckpt_dir, "params.json"), 'r') as f:
            params = json.loads(f.read())
            
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        )
        tokenizer = Tokenizer(model_path=codellama_tokenizer)
        model_args.vocab_size = tokenizer.n_words
        codellama = Transformer(model_args)

        # Load weights
        ckpts = sorted(Path(codellama_ckpt_dir).glob("*.pth"))
        for ckpt in ckpts:
            ckpt_data = torch.load(ckpt, map_location="cpu")
            codellama.load_state_dict(ckpt_data, strict=False)

        return codellama.to(device), tokenizer

    def _load_repairllama(self, repairllama_model_dir, repairllama_lora_dir, register_attention_hooks=True):
        tokenizer = AutoTokenizer.from_pretrained(repairllama_model_dir, trust_remote_code=True)

        repairllama = AutoModelForCausalLM.from_pretrained(
            repairllama_model_dir,
            torch_dtype=torch.float16,
            # load_in_8bit=True, # commented initially
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0),
        )

        repairllama = PeftModel.from_pretrained(repairllama, repairllama_lora_dir, torch_dtype=torch.float16)
        repairllama.config.pad_token = tokenizer.pad_token = tokenizer.unk_token

        if register_attention_hooks:
            for layer_id, layer in enumerate(repairllama.model.model.layers):
                layer.layer_id = layer_id
                layer.register_forward_hook(self._hook_fn)

        return repairllama.to(device), tokenizer

    def _hook_fn(self, module, input, output):
        layer_id = module.layer_id
        self.attention_hooks_data[layer_id] = {"input": tuple(inp.detach() for inp in input)}

    def forward(self, repairllama_input_ids, codellama_input_ids, repairllama_labels, codellama_labels):
        batch_size = repairllama_input_ids.size(0)
        assert repairllama_input_ids.size() == codellama_input_ids.size(), "Input batch sizes must match."

        repairllama_input_ids = repairllama_input_ids.to(device)
        codellama_input_ids = codellama_input_ids.to(device)

        repairllama_hidden_states = self.repairllama.get_input_embeddings()(repairllama_input_ids)
        codellama_hidden_states = self.codellama.tok_embeddings(codellama_input_ids)

        for i in range(len(self.repairllama.model.model.layers)):
            repairllama_hidden_states = self.repairllama.model.model.layers[i](
                hidden_states=repairllama_hidden_states, attention_mask=None
            )

            assert i in self.attention_hooks_data, f"Attention hooks missing for layer {i}"
            dynamic_adapter = self.attention_hooks_data[i]["input"][0]
            codellama_hidden_states = self.codellama.layers[i](
                codellama_hidden_states, None, None, None, dynamic_adapter
            )

        repairllama_logits = self.repairllama.lm_head(repairllama_hidden_states)
        codellama_logits = self.codellama.lm_head(codellama_hidden_states)

        repairllama_loss = self.criterion(repairllama_logits.view(-1, repairllama_logits.size(-1)), repairllama_labels.view(-1))
        codellama_loss = self.criterion(codellama_logits.view(-1, codellama_logits.size(-1)), codellama_labels.view(-1))

        return repairllama_loss, codellama_loss
    
    @torch.inference_mode()
    def forward_inference(self, repairllama_input_ids, codellama_input_ids, start_pos: int, adaptor=False):
        # Ensure batch size consistency
        assert repairllama_input_ids.shape[0] == codellama_input_ids.shape[0] # batch_size should be equal

        # Move inputs to the correct device
        repairllama_input_ids = repairllama_input_ids.to(device)
        if adaptor:
            codellama_input_ids = codellama_input_ids.to(device)

        # RepairLlama configuration
        _bsz, repairllama_seqlen = repairllama_input_ids.shape
        repairllama_h = self.repairllama.model.model.embed_tokens(repairllama_input_ids)

        # RepairLlama position ids and mask
        repairllama_position_ids = torch.arange(
            repairllama_seqlen, dtype=torch.long, device=repairllama_input_ids.device
        ).unsqueeze(0).expand(_bsz, -1)

        repairllama_mask = torch.full(
            (1, 1, repairllama_seqlen, repairllama_seqlen),
            float("-inf"),
            device=repairllama_h.device
        )
        repairllama_mask = torch.triu(repairllama_mask, diagonal=1).type_as(repairllama_h)

        # CodeLlama configuration (if adapter is enabled)
        if adaptor:
            _bsz, codellama_seqlen = codellama_input_ids.shape
            codellama_h = self.codellama.tok_embeddings(codellama_input_ids)
            codellama_freq_cis = self.codellama.freqs_cis.to(codellama_h.device)[:codellama_seqlen]
            codellama_mask = torch.full(
                (1, 1, codellama_seqlen, codellama_seqlen),
                float("-inf"),
                device=codellama_h.device
            )
            codellama_mask = torch.triu(codellama_mask, diagonal=1).type_as(repairllama_h)

        # Ensure layer consistency
        assert self.repairllama.config.num_hidden_layers == self.codellama.config['num_hidden_layers']
        n_layers = self.repairllama.config.num_hidden_layers

        # Process through layers
        for i in range(n_layers):
            repairllama_h, *_ = self.repairllama.model.model.layers[i](
                repairllama_h, repairllama_mask, repairllama_position_ids
            )
            assert self.attention_hooks_data.get(i) is not None, f"Missing attention data for layer {i}."

            if adaptor:
                dynamic_adaptor = self.attention_hooks_data[i].get('input')[0]
                codellama_h = self.codellama.layers[i](
                    codellama_h, start_pos, codellama_freq_cis, codellama_mask, dynamic_adaptor
                )

        # Reset attention hooks data
        self.attention_hooks_data = {}

        # Process RepairLlama output
        repairllama_h = self.repairllama.model.model.norm(repairllama_h[0])
        repairllama_output = self.repairllama.model.lm_head(repairllama_h)

        # Process CodeLlama output (if adapter is enabled)
        if adaptor:
            codellama_h = self.codellama.norm(codellama_h)
            codellama_output = self.codellama.output(codellama_h[:, -1, :])
        else:
            codellama_output = None
        return repairllama_output, codellama_output.float() if codellama_output is not None else None
    
    @torch.inference_mode()
    def generate(self, repairllama_input_ids, codellama_input_ids=None,
                max_gen_len: int = 256, temperature: float = 0.1,
                top_p: float = 0.75):
        bsz = len(repairllama_input_ids)

        # Initialize `codellama_input_ids` if not provided
        if codellama_input_ids is None:
            codellama_input_ids = torch.full(
                (bsz, 1), 
                fill_value=0,  # Using 0 as padding token for testing
                dtype=torch.long
            )

        # Ensure batch sizes match
        assert len(repairllama_input_ids) == len(codellama_input_ids), "Batch sizes must match."

        # Unified parameter check
        params = self.codellama.params
        assert bsz <= params.max_batch_size, f"Batch size {bsz} exceeds the maximum allowed {params.max_batch_size}."

        # Handle input strings
        if isinstance(repairllama_input_ids[0], str):
            repairllama_input_ids = [
                self.repairllama_tokenizer.encode(x, return_tensors='pt') for x in repairllama_input_ids
            ]

        if isinstance(codellama_input_ids[0], str):
            codellama_input_ids = [
                self.codellama_tokenizer.encode(x, bos=True, eos=False) for x in codellama_input_ids
            ]

        # Calculate prompt sizes
        min_repairllama_prompt_size = min(len(t) for t in repairllama_input_ids)
        max_repairllama_prompt_size = max(len(t) for t in repairllama_input_ids)
        min_codellama_prompt_size = min(len(t) for t in codellama_input_ids)
        max_codellama_prompt_size = max(len(t) for t in codellama_input_ids)

        # Prepare token tensors
        total_repairllama_len = min(params.max_seq_len, max_gen_len + max_repairllama_prompt_size)
        repairllama_tokens = torch.full((bsz, total_repairllama_len), self.repairllama_tokenizer.pad_token_id).cuda().long()

        total_codellama_len = min(params.max_seq_len, max_gen_len + max_codellama_prompt_size)
        codellama_tokens = torch.full((bsz, total_codellama_len), 0).cuda().long()

        for k, t in enumerate(repairllama_input_ids):
            repairllama_tokens[k, :len(t)] = torch.tensor(t).cuda().long()

        input_repairllama_text_mask = repairllama_tokens != self.repairllama_tokenizer.pad_token_id
        repairllama_start_pos = min_repairllama_prompt_size

        for k, t in enumerate(codellama_input_ids):
            codellama_tokens[k, :len(t)] = torch.tensor(t).cuda().long()

        input_codellama_text_mask = codellama_tokens != 0
        codellama_start_pos = max(max_repairllama_prompt_size, total_repairllama_len - total_codellama_len)

        # Iterative token generation
        prev_pos = 0
        for cur_pos in range(repairllama_start_pos, total_repairllama_len):
            with torch.cuda.amp.autocast():
                if cur_pos < codellama_start_pos:
                    repairllama_output, _ = self.forward_inference(
                        repairllama_tokens[:, prev_pos:cur_pos], None, prev_pos
                    )
                else:
                    repairllama_output, codellama_logits = self.forward_inference(
                        repairllama_tokens[:, prev_pos:cur_pos], codellama_tokens[:, :cur_pos], prev_pos, adaptor=True
                    )

            next_repairllama_token = repairllama_output[:, -1].argmax(dim=-1)
            next_repairllama_token = torch.where(
                input_repairllama_text_mask[:, cur_pos],
                repairllama_tokens[:, cur_pos],
                next_repairllama_token
            )
            repairllama_tokens[:, cur_pos] = next_repairllama_token

            codellama_cur_pos = cur_pos - codellama_start_pos
            if codellama_cur_pos >= 0:
                next_codellama_token = codellama_logits[:, -1].argmax(dim=-1)
                next_codellama_token = torch.where(
                    input_codellama_text_mask[:, codellama_cur_pos],
                    codellama_tokens[:, codellama_cur_pos],
                    next_codellama_token
                )
                codellama_tokens[:, codellama_cur_pos] = next_codellama_token

                if bsz == 1 and next_codellama_token[0] == self.codellama_tokenizer.eos_id:
                    break

            prev_pos = cur_pos

        # Decode generated tokens
        repairllama_decoded = [
            self.repairllama_tokenizer.decode(
                t[len(repairllama_input_ids[i]): len(repairllama_input_ids[i]) + max_gen_len], skip_special_tokens=True
            ) for i, t in enumerate(repairllama_tokens.tolist())
        ]

        codellama_decoded = [
            self.codellama_tokenizer.decode(
                t[len(codellama_input_ids[i]): len(codellama_input_ids[i]) + max_gen_len], skip_special_tokens=True
            ) for i, t in enumerate(codellama_tokens.tolist())
        ]

        return repairllama_decoded, codellama_decoded