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
        codellama_ckpt_dir=args.codellama_ckpt_dir,
        codellama_tokenizer=args.codellama_tokenizer_path,
        repairllama_model_dir=args.repairllama_model_dir or 'codellama/CodeLlama-7b-hf',
        repairllama_lora_dir=args.repairllama_lora_dir or './repairllama-lora',
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        w_lora=args.w_lora,
        lora_rank=args.lora_rank
    )
    llama_adapter.load_codellma_tuned(args.codellama_trained_weight_dir)
    
    # repairllama_input_ids = torch.load(f"{args.repairllama_input_pth}",  map_location=torch.device('cpu'))
    df = pd.read_csv(args.repairllama_input_pth)
    repairllama_input = df["buggy_code"].tolist()

    codellama_input_ids = torch.load(args.codellama_input_pth) if args.codellama_input_pth is not None else None # commented for testing
    # codellama_input_ids=[128000,    791,   4113,   2082,  44447,   6880,   1595,   6236,  55358,
    #        4619,    315,   1595,   2527,  55358,    389,    279,   4617,     11,
    #         902,  52535,    279,   1595,   6236,  55358,   1749,    389,    279,
    #        1510,   4617,   4856,   1109,   6041,    264,    502,   4617,     13,
    #         578,   8521,   2082,  41800,   1595,   6236,  55358,    449,   1595,
    #        2527,    368,   7964,  10923,    279,   1595,  66665,  26512,     63,
    #         311,   1629,    304,   1202,   1866,   4617,    439,  10825,     13,
    #        1115,   2349,  26420,    430,    279,  23732,    374,  10273,  31978,
    #         304,    264,   8821,   4617,     11,  18899,    279, 100039,    323,
    #       15293,    315,    279,  31009,  11850,   1887,     13, 128001]
    

    # Prepare a file to write the outputs
    output_file = "generated_outputs.txt"
    # print(repairllama_input[70:72])
    # Run forward inference and save outputs
    with torch.no_grad():
        print("Running generate...")
        
        all_repairllama_outputs = []
        all_codellama_outputs = []
        # Process each input ID in repairllama_input_ids
        for i, repair_input in enumerate(repairllama_input[:1]):
            print(f"Processing record {i + 1}/{len(repairllama_input[:1])}...")
            # print("repairllama_input", repair_input)
            # print("codellama_input_ids", codellama_input_ids)
            
            # Generate outputs for the current input
            repairllama_outputs, codellama_outputs = llama_adapter.generate_2(
                repairllama_input_ids=[repair_input],  # Process single input at a time
                codellama_input_ids=codellama_input_ids
            )

            # Collect outputs
            all_repairllama_outputs.append(repairllama_outputs)
            all_codellama_outputs.append(codellama_outputs)

        # Write all outputs to a file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Repairllama Outputs:\n")
            for i, output in enumerate(all_repairllama_outputs):
                f.write(f"Record {i + 1}:\n{output}\n\n")

            f.write("\nCodellama Outputs:\n")
            for i, output in enumerate(all_codellama_outputs):
                f.write(f"Record {i + 1}:\n{output}\n\n")

    print(f"All outputs saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass configuration paths.")
    parser.add_argument("--codellama_ckpt_dir", type=str, required=True, help="Path to CodeLlama checkpoint directory")
    parser.add_argument("--codellama_tokenizer_path", type=str, required=True, help="Path to CodeLlama tokenizer")
    parser.add_argument("--codellama_trained_weight_dir", type=str, required=True, help="Path to trained weights")
    parser.add_argument("--repairllama_input_pth", type=str, required=True, help="RepairLLama input for forward inference")
    parser.add_argument("--codellama_input_pth", type=str, required=False, help="CodeLLAma input for forward inference")
    parser.add_argument("--repairllama_model_dir", type=str, required=False, help="Path to RepairLlama model directory")
    parser.add_argument("--repairllama_lora_dir", type=str, required=False, help="Path to RepairLlama LoRA directory")
    parser.add_argument("--max_batch_size", type=int, required=True, help="Max batch size")
    parser.add_argument('--max_seq_len', default=512, type=int, help='max number of input words')
    parser.add_argument('--w_lora', default=False, type=bool)
    parser.add_argument('--lora_rank', default=16, type=int, help='This only apply if the w_lora parameter is "True"')
    
    args = parser.parse_args()
    main(args)