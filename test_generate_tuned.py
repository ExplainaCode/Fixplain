from .adapter import LLamaAdapter
import argparse
import torch

def main(args):

    llama_adapter = LLamaAdapter(
        codellama_ckpt_dir=args.codellama_ckpt_dir,
        codellama_tokenizer=args.codellama_tokenizer_path,
        repairllama_model_dir=args.repairllama_model_dir or 'codellama/CodeLlama-7b-hf',
        repairllama_lora_dir=args.repairllama_lora_dir or './repairllama-lora',
        max_seq_len=512,
        max_batch_size=args.max_batch_size,
    )
    
    repairllama_input_ids = torch.load(f"{args.repairllama_input_pth}",  map_location=torch.device('cpu'))
    codellama_input_ids = torch.load(args.codellama_input_pth) if args.codellama_input_pth is not None else None
    

    # Prepare a file to write the outputs
    output_file = "generated_outputs.txt"
    print(repairllama_input_ids[70])
    # Run forward inference and save outputs
    with torch.no_grad():
        print("Running generate...")
        
        all_repairllama_outputs = []
        all_codellama_outputs = []
        # Process each input ID in repairllama_input_ids
        for i, repair_input in enumerate(repairllama_input_ids[70:75]):
            print(f"Processing record {i + 1}/{len(repairllama_input_ids[70:75])}...")
            
            # Generate outputs for the current input
            repairllama_outputs, codellama_outputs = llama_adapter.generate(
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
    parser.add_argument("--repairllama_input_pth", type=str, required=True, help="RepairLLama input for forward inference")
    parser.add_argument("--codellama_input_pth", type=str, required=False, help="CodeLLAma input for forward inference")
    parser.add_argument("--repairllama_model_dir", type=str, required=False, help="Path to RepairLlama model directory")
    parser.add_argument("--repairllama_lora_dir", type=str, required=False, help="Path to RepairLlama LoRA directory")
    parser.add_argument("--max_batch_size", type=int, required=True, help="Max batch size")
    
    args = parser.parse_args()
    main(args)