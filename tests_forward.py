from .adapter import LLamaAdapter
import argparse
import torch
from torch.nn.utils.rnn import pad_sequence

def pad_and_mask(inputs, pad_token=0):
    # Convert to tensors if not already
    tensors = [torch.tensor(seq) for seq in inputs]
    # Pad sequences to the length of the longest sequence
    padded = pad_sequence(tensors, batch_first=True, padding_value=pad_token)
    # Create attention masks: 1 for real tokens, 0 for padding
    # attention_masks = (padded != pad_token).long()
    return padded

def main(args):

    llama_adapter = LLamaAdapter(
        codellama_ckpt_dir=args.codellama_ckpt_dir,
        codellama_tokenizer=args.codellama_tokenizer_path,
        repairllama_model_dir=args.repairllama_model_dir or 'codellama/CodeLlama-7b-hf',
        repairllama_lora_dir=args.repairllama_lora_dir or './repairllama-lora',
        max_seq_len=512,
        max_batch_size=args.max_batch_size,
    )

    # Load sequences
    repairllama_input_ids1 = torch.flatten(torch.load(f"{args.repairllama_input_pth}1.pt"))
    repairllama_input_ids2 = torch.flatten(torch.load(f"{args.repairllama_input_pth}2.pt"))
    codellama_labels1 = torch.flatten(torch.load(f"{args.codellama_labels_pth}1.pt"))
    codellama_labels2 = torch.flatten(torch.load(f"{args.codellama_labels_pth}2.pt"))

    # Combine sequences into a list for batching
    repairllama_inputs = [repairllama_input_ids1, repairllama_input_ids2]
    codellama_labels = [codellama_labels1, codellama_labels2]

    # Pad and create attention masks
    repairllama_padded = pad_and_mask(repairllama_inputs)
    codellama_padded = pad_and_mask(codellama_labels)

    # Run forward_inference
    with torch.no_grad():
        print("Running generate...")
        codellama_c_loss, next_repairllama_cache = llama_adapter.forward(
            repairllama_input_ids=repairllama_padded,
            codellama_input_ids=None,
            repairllama_labels=None,
            codellama_labels=codellama_padded
        )
        print("codellama loss: \n", codellama_c_loss)


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