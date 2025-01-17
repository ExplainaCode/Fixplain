from .adapter import LLamaAdapter
import argparse
import torch

def main(args):
    # print("CodeLlama Checkpoint Directory:", args.codellama_ckpt_dir)
    # print("CodeLlama Tokenizer Path:", args.codellama_tokenizer_path)
    # print("RepairLlama Model Directory:", args.repairllama_model_dir)
    # print("RepairLlama LoRA Directory:", args.repairllama_lora_dir)

    llama_adapter = LLamaAdapter(
        codellama_ckpt_dir=args.codellama_ckpt_dir,
        codellama_tokenizer=args.codellama_tokenizer_path,
        repairllama_model_dir=args.repairllama_model_dir or 'codellama/CodeLlama-7b-hf',
        repairllama_lora_dir=args.repairllama_lora_dir or './repairllama-lora',
        max_seq_len=512,
        max_batch_size=args.max_batch_size,
    )
    
    repairllama_input_ids = torch.load(f"{args.repairllama_input_pth}")
    codellama_input_ids = torch.load(args.codellama_input_pth) if args.codellama_input_pth is not None else None
    
    # Run forward_inference
    with torch.no_grad():
        print("Running generate...")
        repairllama_outputs, codellama_outputs = llama_adapter.generate(
            repairllama_input_ids=[repairllama_input_ids[0], repairllama_input_ids[1]], codellama_input_ids=codellama_input_ids
        )
        print("Repairllama: \n", repairllama_outputs)
        print("Codellama: \n", codellama_outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass configuration paths.")
    parser.add_argument("--codellama_ckpt_dir", type=str, required=True, help="Path to CodeLlama checkpoint directory")
    parser.add_argument("--codellama_tokenizer_path", type=str, required=True, help="Path to CodeLlama tokenizer")
    parser.add_argument("--repairllama_input_pth", type=str, required=True, help="RepairLLama input for forward inference")
    parser.add_argument("--codellama_input_pth", type=str, required=False, help="CodeLLAma input for forward inference")
    parser.add_argument("--repairllama_model_dir", type=str, required=False, help="Path to RepairLlama model directory")
    parser.add_argument("--repairllama_lora_dir", type=str, required=False, help="Path to RepairLlama LoRA directory")
    parser.add_argument("--max_batch_size", type=int, required=True, help="Max batch size")
    
    args = parser.parse_args()
    main(args)