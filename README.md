# Two-Stream LLaMA for XLNet-style Permutation Language Modeling

This repository implements a two-stream attention mechanism on top of LLaMA, inspired by XLNet's permutation language modeling approach. The model supports flexible grouping strategies for predicting tokens in arbitrary orders rather than the conventional left-to-right autoregressive approach.

## Key Features

### 1. Two-Stream Architecture
- **Content Stream**: Uses input embeddings for Q/K/V, can see tokens in groups ≤ current group
- **Query Stream**: Uses learnable query vectors for Q, K/V from content stream, can only see tokens in groups < current group

### 2. Flexible Grouping System
- **Default**: Conventional autoregressive (one token per group, left-to-right)
- **Custom**: Arbitrary group sizes and orders with automatic mask generation
- **Automatic**: Masks are generated automatically from group IDs

### 3. FlashAttention 2 Support
- Optimized for bidirectional attention
- Efficient memory usage
- Compatible with modern GPU architectures

## Architecture Overview

```
Input Tokens → Embeddings → Content Stream (Q/K/V from content)
                    ↓
              Query Embed → Query Stream (Q from query, K/V from content)
                    ↓
              Two-Stream Decoder Layers
                    ↓
              Output (Content stream for final predictions)
```

## Usage Examples

### Basic Usage

```python
from llama_xlnet import LlamaTwoStreamForCausalLM
from transformers import LlamaConfig

# Create model with FlashAttention 2
config = LlamaConfig(
    _attn_implementation="flash_attention_2",  # Important!
    # ... other config parameters
)
model = LlamaTwoStreamForCausalLM(config)

# Default behavior: conventional autoregressive
outputs = model(input_ids=input_ids)
```

### Custom Grouping

```python
import torch

# Define custom grouping: tokens 0,1 in group 0; tokens 2,3 in group 1; token 4 in group 2
group_ids = torch.tensor([[0, 0, 1, 1, 2]])

# This means:
# - Group 0: tokens 0,1 can see each other and are predicted together
# - Group 1: tokens 2,3 can see groups 0 and themselves, predicted together
# - Group 2: token 4 can see all previous groups, predicted alone

# Use with model
outputs = model(input_ids=input_ids, group_ids=group_ids)
```

### Using the set_grouping Method

```python
# Set grouping on the model
model.set_grouping(group_ids)

# Now you can call without passing group_ids each time
outputs = model(input_ids=input_ids)  # group_ids automatically used
```

## Grouping Patterns

### 1. Conventional Autoregressive (Default)
```python
group_ids = torch.tensor([[0, 1, 2, 3, 4]])  # Each token is its own group
```

### 2. Pair Prediction
```python
group_ids = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]])  # Predict tokens in pairs
```

### 3. Half-and-Half
```python
group_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])  # Predict first half, then second half
```

### 4. Custom Patterns
```python
# Any non-decreasing sequence of group IDs
group_ids = torch.tensor([[0, 0, 1, 2, 2, 3, 4, 4]])
```

## Attention Mask Behavior

### Content Stream
- Can attend to tokens in groups ≤ current group
- Enables bidirectional attention within the current group
- Useful for capturing local context

### Query Stream  
- Can only attend to tokens in groups < current group
- Prevents information leakage from future groups
- Maintains causal structure for prediction

## Implementation Details

### Mask Generation
The `generate_grouping_masks()` function automatically creates:
- **Content mask**: `(group_i >= group_j)` - can see current and previous groups
- **Query mask**: `(group_i > group_j)` - can only see previous groups
- **Target mapping**: Maps each position to its target group

### Validation
- Group IDs must be non-decreasing (groups are ordered)
- Currently supports batch_size=1 for grouping
- Automatic fallback to conventional autoregressive if no grouping specified

## Requirements

- PyTorch >= 2.0
- FlashAttention 2
- Transformers library
- CUDA-compatible GPU (recommended)

## Installation

```bash
pip install torch transformers flash-attn
```

## Example Script

Run the included example to see the grouping system in action:

```bash
python example_usage.py
```

## Notes

- The model automatically generates appropriate attention masks from group IDs
- FlashAttention 2 is required for bidirectional attention support
- The "eager" and SDPA attention implementations are disabled for bidirectional use
- Cache support is limited in the current implementation

## Future Enhancements

- Support for batch_size > 1 in grouping
- More sophisticated permutation strategies
- Enhanced cache handling for generation
- Support for other attention implementations 