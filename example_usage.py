#!/usr/bin/env python3
"""
Example usage of LlamaTwoStreamForCausalLM with custom grouping for permutation language modeling.
"""

import torch
from transformers import AutoTokenizer
from llama_xlnet import LlamaTwoStreamForCausalLM, generate_grouping_masks, generate_default_grouping

def main():
    # Example 1: Conventional autoregressive (default behavior)
    print("=== Example 1: Conventional Autoregressive ===")
    
    # Create a simple model (you would normally load a pretrained one)
    from transformers import LlamaConfig
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=2048,
        _attn_implementation="flash_attention_2"  # Important for bidirectional attention
    )
    
    model = LlamaTwoStreamForCausalLM(config)
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Example text
    text = "Hello world, how are you today?"
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    # Default behavior: conventional autoregressive (each token is its own group)
    print(f"Input text: {text}")
    print(f"Input tokens: {tokenizer.convert_ids_to_tokens(inputs.input_ids[0])}")
    
    # The model will automatically generate default grouping
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Output shape: {outputs.logits.shape}")
    print()
    
    # Example 2: Custom grouping
    print("=== Example 2: Custom Grouping ===")
    
    # Define custom grouping: tokens 0,1 in group 0; tokens 2,3 in group 1; token 4 in group 2
    # This means:
    # - Group 0: tokens 0,1 can see each other and are predicted together
    # - Group 1: tokens 2,3 can see groups 0 and themselves, predicted together  
    # - Group 2: token 4 can see all previous groups, predicted alone
    custom_group_ids = torch.tensor([[0, 0, 1, 1, 2]])
    
    print(f"Custom group IDs: {custom_group_ids[0].tolist()}")
    print(f"Group 0: tokens 0,1 (can see each other)")
    print(f"Group 1: tokens 2,3 (can see groups 0 and themselves)")
    print(f"Group 2: token 4 (can see all previous groups)")
    
    # Generate masks manually to see what they look like
    content_mask, query_mask, target_mapping = generate_grouping_masks(
        custom_group_ids, 
        seq_len=5, 
        device=torch.device('cpu')
    )
    
    print(f"\nContent stream attention mask (can see groups <= current group):")
    print(content_mask[0])  # Show first batch
    
    print(f"\nQuery stream attention mask (can only see groups < current group):")
    print(query_mask[0])  # Show first batch
    
    print(f"\nTarget mapping:")
    print(target_mapping[0])  # Show first batch
    
    # Use the custom grouping with the model
    with torch.no_grad():
        outputs = model(**inputs, group_ids=custom_group_ids)
    
    print(f"\nOutput shape with custom grouping: {outputs.logits.shape}")
    print()
    
    # Example 3: Using the set_grouping method
    print("=== Example 3: Using set_grouping Method ===")
    
    # Set grouping on the model
    model.set_grouping(custom_group_ids)
    
    # Now you can call the model without passing group_ids each time
    with torch.no_grad():
        outputs = model(**inputs)  # group_ids will be automatically used
    
    print(f"Output shape after set_grouping: {outputs.logits.shape}")
    print()
    
    # Example 4: Different grouping patterns
    print("=== Example 4: Different Grouping Patterns ===")
    
    # Pattern 1: Predict tokens in pairs
    pair_group_ids = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]])
    print(f"Pair grouping: {pair_group_ids[0].tolist()}")
    
    # Pattern 2: Predict first half, then second half
    half_group_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    print(f"Half grouping: {half_group_ids[0].tolist()}")
    
    # Pattern 3: Predict every other token first
    skip_group_ids = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]])
    print(f"Skip grouping: {skip_group_ids[0].tolist()}")
    
    # Note: The skip pattern above would violate the non-decreasing constraint
    # and would raise an error if used with the model

if __name__ == "__main__":
    main() 