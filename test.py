from .adapter import LLamaAdapter
import argparse
import torch

def main(args):
    print("CodeLlama Checkpoint Directory:", args.codellama_ckpt_dir)
    print("CodeLlama Tokenizer Path:", args.codellama_tokenizer_path)
    print("RepairLlama Model Directory:", args.repairllama_model_dir)
    print("RepairLlama LoRA Directory:", args.repairllama_lora_dir)

    llama_adapter = LLamaAdapter(
        codellama_ckpt_dir=args.codellama_ckpt_dir,
        codellama_tokenizer=args.codellama_tokenizer_path,
        repairllama_model_dir=args.repairllama_model_dir or 'codellama/CodeLlama-7b-hf',
        repairllama_lora_dir=args.repairllama_lora_dir or './repairllama-lora',
        max_seq_len=512,
        max_batch_size=1,
        phase="finetune",
        w_lora=True
    )

    # Create dummy inputs
    batch_size = 1
    seq_len = 128
    vocab_size = 32000  # Adjust as per the tokenizer used
    
    repairllama_input_ids = torch.load(args.repairllama_input_ids)
    codellama_input_ids = torch.load(args.codellama_input_ids)

    start_pos = 0

    # Run forward_inference
    with torch.no_grad():
        print("Running forward inference...")
        outputs = llama_adapter.forward_inference(
            repairllama_input_ids=repairllama_input_ids,
            codellama_input_ids=codellama_input_ids,
            start_pos=start_pos,
            adapter=True,
        )
        print("Outputs:", outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass configuration paths.")
    parser.add_argument("--codellama_ckpt_dir", type=str, required=True, help="Path to CodeLlama checkpoint directory")
    parser.add_argument("--codellama_tokenizer_path", type=str, required=True, help="Path to CodeLlama tokenizer")
    parser.add_argument("--repairllama_input_ids", type=str, required=True, help="RepairLLama input for forward inference")
    parser.add_argument("--codellama_input_ids", type=str, required=True, help="CodeLLAma input for forward inference")
    parser.add_argument("--repairllama_model_dir", type=str, required=False, help="Path to RepairLlama model directory")
    parser.add_argument("--repairllama_lora_dir", type=str, required=False, help="Path to RepairLlama LoRA directory")
    
    args = parser.parse_args()
    main(args)