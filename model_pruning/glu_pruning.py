import torch
from transformers import LlavaForConditionalGeneration, AutoTokenizer
from torch import nn


def compute_neuron_pair_importance(gate_weight, up_weight):
    """
    compute neuron pair importance scores (Maximum Absolute Weight)

    Args:
    - gate_weight: Weight matrix from the gate_proj layer.
    - up_weight: Weight matrix from the up_weight layer.

    Returns:
    - importance_scores: Importance scores for each neuron pair.
    """

    gate_max_abs = torch.max(gate_weight, dim=1).values + torch.abs(
        torch.min(gate_weight, dim=1).values
    )
    up_max_abs = torch.max(up_weight, dim=1).values + torch.abs(
        torch.min(up_weight, dim=1).values
    )
    importance_scores = gate_max_abs + up_max_abs
    return importance_scores


def prune_neuron_pairs(mlp, prune_percent):
    """
    Reduces the dimensions of the **gate_proj**,**up_proj**, **down_proj**
    layers removing the least important neurons.

    Args:
    - mlp: Layers to prune.
    - prune_percent: Percentage of neurons to prune.

    Returns:
    - new_gate_proj, new_up_proj, new_down_proj:  New pruned layers.
    - k: New intermediate size.

    """
    # Extract the weights from the MLP layers
    #  these weights are used to calculate each neuron's
    #  importance score in the next step.
    gate_weight = mlp.gate_proj.weight.data.float()
    up_weight = mlp.up_proj.weight.data.float()

    # Compute importance stores. Neurons with higher importance scores
    # are considered more important and less likely to be pruned.
    importance_scores = compute_neuron_pair_importance(gate_weight, up_weight)

    # Store the original number of neurons in the intermediate layer.
    original_intermediate_size = gate_weight.size(0)
    # Computes the number of neurons to prune.
    num_neuron_pairs_to_prune = min(
        int(prune_percent * original_intermediate_size), original_intermediate_size - 1
    )
    # Calculate the number of neurons to keep. The new intermediate size.
    k = original_intermediate_size - num_neuron_pairs_to_prune

    # Just check that there is no big error calculating k. We can't prune all the neurons.
    if k <= 0:
        raise ValueError(
            f"Invalid number of neuron pairs to keep: {k}. Adjust the prune_percent."
        )

    _, indices_to_keep = torch.topk(importance_scores, k, largest=True, sorted=True)
    indices_to_keep = indices_to_keep.sort().values

    # create the new layers
    new_gate_proj = nn.Linear(mlp.gate_proj.in_features, k, bias=False).to("cuda")
    new_up_proj = nn.Linear(mlp.up_proj.in_features, k, bias=False).to("cuda")
    new_down_proj = nn.Linear(k, mlp.down_proj.out_features, bias=False).to("cuda")

    # copy weights to the new layers.
    new_gate_proj.weight.data = mlp.gate_proj.weight.data[indices_to_keep, :]
    new_up_proj.weight.data = mlp.up_proj.weight.data[indices_to_keep, :]
    new_down_proj.weight.data = mlp.down_proj.weight.data[:, indices_to_keep]

    # return new layers and intermediate size.
    return new_gate_proj, new_up_proj, new_down_proj, k


def update_model(model, prune_percent):
    """
    It modifies each mlp layer present in model, to retain only the most
    important neurons. Creating new smaller versions of each layer pruned.

    Args:
    - model: Model to prune.
    - prune_percent: Percentage of neurons to prune.

    Returns:
    - model: New pruned model.
    """
    new_intermediate_size = None

    # loop for each model layer.
    for idx, layer in enumerate(model.layers):
        # Since each layer is a LlamaDecoderLayer it contains multiple components
        # Attention, MLP and Layer norms. We're targetting MLP component
        # by accesing layer.mlp.
        mlp = layer.mlp

        # Call the prune_neiron_pairs with the layers and receiving the pruned.
        new_gate_proj, new_up_proj, new_down_proj, new_size = prune_neuron_pairs(
            mlp, prune_percent
        )

        # Replace the Origiginal Layers with Pruned Layers.
        mlp.gate_proj = new_gate_proj
        mlp.up_proj = new_up_proj
        mlp.down_proj = new_down_proj

        # new_intermediate_size only needs to be set once
        if new_intermediate_size is None:
            new_intermediate_size = new_size

    # Update the model config file.
    model.config.intermediate_size = new_intermediate_size

    return model


if __name__ == "__main__":
    model_id = "llava-hf/llava-1.5-7b-hf"
    pruned_model_name = "llava-1.5-7b-hf-glu-pruned-70pct"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16
    ).to("cuda")

    update_model(model.language_model, prune_percent=0.7)

    # Update the root model config to match the pruned language model config
    # This is crucial because Qwen2VLForConditionalGeneration's config needs to reflect the change
    # for correct loading.
    if hasattr(model.language_model, "config") and hasattr(
        model.language_model.config, "intermediate_size"
    ):
        model.config.intermediate_size = model.language_model.config.intermediate_size
        # Also update text_config if it exists in the root config, just in case
        if (
            hasattr(model.config, "text_config")
            and model.config.text_config is not None
        ):
            model.config.text_config.intermediate_size = (
                model.language_model.config.intermediate_size
            )

    print(f"Pruning complete. New intermediate_size: {model.config.intermediate_size}")

    model.save_pretrained(pruned_model_name, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(pruned_model_name)
