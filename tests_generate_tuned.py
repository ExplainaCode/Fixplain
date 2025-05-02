from .adapter import LLamaAdapter
import argparse
import torch
import pandas as pd
import os
import torch.distributed as dist
from fairscale.nn.model_parallel import initialize as fs_init

import socket

def find_free_port():
    """Find a free port on the machine"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))            # 0 means select a free port
        return s.getsockname()[1]  # Return the port number

os.environ['MASTER_ADDR'] = 'localhost'   # Use the address of the machine, for single node use 'localhost'
os.environ['MASTER_PORT'] = str(find_free_port())

# Initialize the process group for distributed training
if not dist.is_initialized():
    dist.init_process_group(backend="nccl", 
                            rank=int(os.getenv('RANK', 0)),   # Get RANK from environment variables
                            world_size=int(os.getenv('WORLD_SIZE', 1)))  # Get WORLD_SIZE from environment variables

# Initialize FairScale model parallel group
fs_init.initialize_model_parallel(model_parallel_size_=1)

def main(args):

    llama_adapter = LLamaAdapter(
        llama_ckpt_dir=args.llama_ckpt_dir,
        llama_tokenizer=args.llama_tokenizer_path,
        repairllama_model_dir=args.repairllama_model_dir or 'codellama/CodeLlama-7b-hf',
        repairllama_lora_dir=args.repairllama_lora_dir or './repairllama-lora',
        max_batch_size=args.max_batch_size,
        w_lora=args.w_lora,
        lora_rank=args.lora_rank,
        phase="inference",
    )
    llama_adapter.load_llma_tuned(args.llama_trained_weight_dir)
    
    # repairllama_input_ids = torch.load(f"{args.repairllama_input_pth}",  map_location=torch.device('cpu'))
    df = pd.read_csv(args.repairllama_input_pth)
    repairllama_input = df["buggy_code"].tolist()

    llama_input_ids = torch.load(args.llama_input_pth) if args.llama_input_pth is not None else None # commente for testing

    # Prepare a file to write the outputs
    output_file = "generated_outputs.txt"
    # print(repairllama_input[70:72])
    # Run forward inference and save outputs
    with torch.no_grad():
        print("Running generate...")
        
        all_repairllama_outputs = []
        all_llama_outputs = []
        # Process each input ID in repairllama_input_ids
        for i, repair_input in enumerate(repairllama_input[:100]):
            print(f"Processing record {i + 1}/{len(repairllama_input[:100])}...")
            # print("repairllama_input", repair_input)
            # print("llama_input_ids", llama_input_ids)
            
            # Generate outputs for the current input
            repairllama_outputs, llama_outputs = llama_adapter.generate(
                repairllama_input_ids=[repair_input],  # Process single input at a time
                llama_input_ids=llama_input_ids
            )

            # Collect outputs
            all_repairllama_outputs.append(repairllama_outputs)
            all_llama_outputs.append(llama_outputs)

        # Write all outputs to a file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Repairllama Outputs:\n")
            for i, output in enumerate(all_repairllama_outputs):
                f.write(f"Record {i + 1}:\n{output}\n\n")

            f.write("\nllama Outputs:\n")
            for i, output in enumerate(all_llama_outputs):
                f.write(f"Record {i + 1}:\n{output}\n\n")

    print(f"All outputs saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass configuration paths.")
    parser.add_argument("--llama_ckpt_dir", type=str, required=True, help="Path to Llama checkpoint directory")
    parser.add_argument("--llama_tokenizer_path", type=str, required=True, help="Path to Llama tokenizer")
    parser.add_argument("--llama_trained_weight_dir", type=str, required=True, help="Path to trained weights")
    parser.add_argument("--repairllama_input_pth", type=str, required=True, help="RepairLLama input for forward inference")
    parser.add_argument("--llama_input_pth", type=str, required=False, help="LLAma input for forward inference")
    parser.add_argument("--repairllama_model_dir", type=str, required=False, help="Path to RepairLlama model directory")
    parser.add_argument("--repairllama_lora_dir", type=str, required=False, help="Path to RepairLlama LoRA directory")
    parser.add_argument("--max_batch_size", type=int, required=True, help="Max batch size")
    parser.add_argument('--max_seq_len', default=512, type=int, help='max number of input words')
    parser.add_argument('--w_lora', default=False, type=bool)
    parser.add_argument('--lora_rank', default=16, type=int, help='This only apply if the w_lora parameter is "True"')
    
    args = parser.parse_args()
    main(args)