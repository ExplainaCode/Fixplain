from .adapter import LLamaAdapter
import argparse
import torch
import pandas as pd
import os
import csv
import torch.distributed as dist
from fairscale.nn.model_parallel import initialize as fs_init
import utils.misc as misc
import utils.lr_sched as lr_sched
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.misc import CustomDistributedSampler
from utils.dataset import FinetuneDataset

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
    device = torch.device(args.device)
    model = LLamaAdapter(
        llama_ckpt_dir=args.llama_ckpt_dir,
        llama_tokenizer=args.llama_tokenizer_path,
        llama_max_seq_len=args.llama_max_input_len,
        max_batch_size=args.max_batch_size,
        w_lora=args.w_lora,
        lora_rank=args.lora_rank,
        phase="inference"
    )

    model.load_codellma_tuned(args.llama_trained_weight_dir)
    model.to(device)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     model_without_ddp = model.module

    dataset_inference = FinetuneDataset(
        model=model, 
        dataframe_path=args.data_path,
        phase="inference"
    )
    print(dataset_inference)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_inference = CustomDistributedSampler(
        dataset_inference, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    print("Sampler_inference = %s" % str(sampler_inference))

    data_loader_inference = torch.utils.data.DataLoader(
        dataset_inference, sampler=sampler_inference,
        batch_size=args.max_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        multiprocessing_context='spawn' 
    )

    # Prepare a file to write the outputs
    output_file = "generated_outputs.csv"

    with torch.no_grad():
        print("Running generate...")
        
        metric_logger = misc.MetricLogger(delimiter="  ")
        print_freq = 1

        all_records = []

        model.eval()
        for data_iter_step, (
            llama_input_ids, llama_mask, explanation) in enumerate(
                metric_logger.log_every(data_loader_inference, print_freq)
        ):
                
            # Generate outputs for the current input
            llama_outputs = model.generate(
                llama_input_ids=llama_input_ids,
                llama_mask=llama_mask, 
                batch_size=args.max_batch_size
            )

            # Collect outputs
            all_records.append({
                "gpt_explanation": str(explanation) if explanation is not None else "",
                "llama_output": llama_outputs
            })

        # Write all outputs to a CSV file
        with open(output_file, mode="w", newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["gpt_explanation", "llama_output"])
            writer.writeheader()
            writer.writerows(all_records)

        print(f"All outputs saved to {output_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass configuration paths.")
    parser.add_argument("--llama_ckpt_dir", type=str, required=True, help="Path to Llama checkpoint directory")
    parser.add_argument("--llama_tokenizer_path", type=str, required=True, help="Path to Llama tokenizer")
    parser.add_argument("--llama_trained_weight_dir", type=str, required=True, help="Path to trained weights")
    parser.add_argument("--data_path", type=str, required=True, help="Data path")
    parser.add_argument("--repairllama_model_dir", type=str, required=False, help="Path to RepairLlama model directory")
    parser.add_argument("--repairllama_lora_dir", type=str, required=False, help="Path to RepairLlama LoRA directory")
    parser.add_argument("--max_batch_size", type=int, required=True, help="Max batch size")
    parser.add_argument('--repairllama_max_input_len', default=512, type=int,
                        help='max number of input words(embeddings) in repairllama')
    parser.add_argument('--llama_max_input_len', default=512, type=int,
                        help='max number of input words(embeddings) in llama')
    parser.add_argument('--w_lora', default=False, type=bool)
    parser.add_argument('--lora_rank', default=16, type=int, help='This only apply if the w_lora parameter is "True"')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    
    args = parser.parse_args()
    main(args)