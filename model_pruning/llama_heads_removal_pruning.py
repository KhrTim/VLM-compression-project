import torch
from torch import nn
import random
import math
from transformers import LlavaForConditionalGeneration


def calculate_head_norms(attn_layer, total_heads, head_dim, num_heads_to_remove):
    """
    Selects head indices to prune based on the L2 norm of the O_proj weights.
    """
    o_proj_weight = attn_layer.o_proj.weight.data

    # 1. Calculate the L2 norm for each head
    head_norms = torch.zeros(total_heads)
    for h in range(total_heads):
        # The O_proj input corresponds to the concatenated head outputs.
        # We slice the input dimension (axis=1 in the [out, in] format)
        start_idx = h * head_dim
        end_idx = (h + 1) * head_dim

        # Slice the weight block corresponding to the h-th head's output
        # O_proj is [out_features, in_features]. We slice the in_features (dim=1)
        # to get the weights coming *out* of head h.
        head_weight_block = o_proj_weight[:, start_idx:end_idx]

        # Calculate the L2 norm of this block
        norm = torch.linalg.norm(head_weight_block).item()
        head_norms[h] = norm

    # 2. Determine the number of heads to remove

    if num_heads_to_remove == 0:
        return {}
    return head_norms


def prune_llama_heads(
    attn_layer,
    heads_to_prune,
    num_heads,
    head_dim,
):
    """
    Structurally removes specific attention heads from a LlamaAttention layer.
    AND prunes the input/output projections to match the new global hidden size.

    Args:
        attn_layer: The LlamaAttention module
        heads_to_prune: List of integers (indices of heads to remove, e.g. [0, 5])
        keep_hidden_indices: Tensor of indices to keep for the global hidden dimension (residual stream).
    """

    # 2. Build the Mask for Keeping Indices (Heads)
    # We are looking for indices in the range [0, 2048]
    all_indices = torch.arange(num_heads * head_dim)
    mask = torch.ones(num_heads * head_dim, dtype=torch.bool)

    for h in heads_to_prune:
        start = h * head_dim
        end = (h + 1) * head_dim
        mask[start:end] = False

    # Get the integer indices we want to KEEP for the heads (inner dimension)
    keep_head_indices = all_indices[mask]

    # print(f"Pruning {len(heads_to_prune)} heads. New head count: {num_heads - len(heads_to_prune)}")

    # 3. Helper to prune a Linear layer
    def prune_linear(layer, indices, axis=0):
        # axis=0: Prune output (rows)
        # axis=1: Prune input (cols)
        new_out = layer.out_features
        new_in = layer.in_features

        if axis == 0:
            new_out = len(indices)
            new_weight = layer.weight.data[indices, :]
        else:
            new_in = len(indices)
            new_weight = layer.weight.data[:, indices]

        new_layer = nn.Linear(new_in, new_out, bias=False)
        new_layer.weight.data = new_weight.to(layer.weight.device)
        return new_layer

    # 4. Prune the Projections

    # Q, K, V:
    # - Prune OUTPUT rows (axis 0) based on kept heads.
    # - Prune INPUT columns (axis 1) based on kept hidden dimensions (residual stream).

    attn_layer.q_proj = prune_linear(attn_layer.q_proj, keep_head_indices, axis=0)
    attn_layer.k_proj = prune_linear(attn_layer.k_proj, keep_head_indices, axis=0)
    attn_layer.v_proj = prune_linear(attn_layer.v_proj, keep_head_indices, axis=0)
    attn_layer.o_proj = prune_linear(attn_layer.o_proj, keep_head_indices, axis=1)

    return attn_layer


def select_random_heads_to_prune(total_heads, prune_percentage):
    """
    Randomly selects indices of heads to remove.
    """
    num_heads_to_remove = math.ceil(total_heads * prune_percentage)

    # Ensure we don't try to prune all heads
    if num_heads_to_remove >= total_heads:
        num_heads_to_remove = total_heads - 1

    all_head_indices = list(range(total_heads))

    # Randomly sample the indices to remove
    heads_to_remove = random.sample(all_head_indices, num_heads_to_remove)

    return heads_to_remove


def calculate_indexes_based_on_average_norm(
    language_model, total_heads_number, head_dim, heads_to_prune_number
):
    """
    Selects head indices to prune based on the average L2 norm of the O_proj weights.
    """
    # 3. Sort by norm (ascending) and select the weakest heads
    # (h, norm) tuple: We sort by the norm (index 1)
    # head_norms.sort(key=lambda x: x[1])

    # Select the indices (index 0) of the weakest heads
    # heads_to_remove = [h for h, norm in head_norms[:num_heads_to_remove]]
    avg_norms = torch.zeros(total_heads_number)
    for layer_idx, layer in enumerate(language_model.layers):
        # 2. Call the function using the constant COUNT (PRUNE_COUNT)
        # We assign the result to a NEW variable: `indices_to_remove`
        head_norms = calculate_head_norms(
            layer.self_attn, total_heads_number, head_dim, heads_to_prune_number
        )
        avg_norms += head_norms
    avg_norms /= len(language_model.layers)
    return torch.topk(avg_norms, heads_to_prune_number, largest=False).indices.tolist()


def prune_llama_heads_by_norm(model, prune_percentage):
    TOTAL_HEADS = model.config.text_config.num_attention_heads
    HEAD_DIM = model.config.text_config.head_dim

    # 1. Define the constant COUNT of heads to remove (The target size)
    PRUNE_COUNT = math.ceil(TOTAL_HEADS * prune_percentage)
    new_num_heads = TOTAL_HEADS - PRUNE_COUNT
    new_hidden_size = new_num_heads * HEAD_DIM
    print(f"Target Config: Heads={new_num_heads}, Hidden Size={new_hidden_size}")

    indices_to_remove = calculate_indexes_based_on_average_norm(
        model.model.language_model, TOTAL_HEADS, HEAD_DIM, PRUNE_COUNT
    )
    print(f"Indices to remove: {indices_to_remove}")

    for layer_idx, layer in enumerate(model.model.language_model.layers):
        # 2. Call the function using the constant COUNT (PRUNE_COUNT)
        # We assign the result to a NEW variable: `indices_to_remove`

        print(f"Pruning Layer {layer_idx}: Removing {len(indices_to_remove)} heads.")

        # 3. Apply the structural pruning function using the list of indices
        try:
            layer.self_attn = prune_llama_heads(
                layer.self_attn,
                indices_to_remove,  # Use the list of indices here
                num_heads=TOTAL_HEADS,
                head_dim=HEAD_DIM,
            )
        except Exception as e:
            print(f"ERROR PRUNING Layer {layer_idx}: {e}")
            break

    model.config.text_config.num_attention_heads = new_num_heads
    model.config.text_config.num_key_value_heads = new_num_heads
    # model.config.text_config.hidden_size = new_hidden_size


if __name__ == "__main__":
    model_id = "llava-hf/llava-1.5-7b-hf"
    # model_id = "../llava-1.5-7b-hf-heads-pruned"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16
    ).to("cuda")
    prune_llama_heads_by_norm(model, 0.3)
