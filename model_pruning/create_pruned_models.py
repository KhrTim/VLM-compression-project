#!/usr/bin/env python3
import torch
import sys
import time
import itertools
from torch import nn
from torch.nn.utils import prune
from transformers import LlavaForConditionalGeneration, AutoTokenizer
from glu_pruning import update_model as glu_prune_model
from llama_heads_removal_pruning import prune_llama_heads_by_norm
import os


def apply_l1_pruning(model, amount=0.3):
    print(
        f"Applying L1 unstructured pruning ({amount * 100}% globally to language_model only)..."
    )

    # Only prune layers in model.model.language_model as requested
    # Process in batches to avoid memory issues
    parameters_to_prune = []
    for name, module in model.model.language_model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, "weight"))

    print(f"Found {len(parameters_to_prune)} linear layers in language_model to prune")

    # Process in smaller batches to manage GPU memory
    batch_size = 5  # Process 10 layers at a time
    total_batches = (len(parameters_to_prune) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(parameters_to_prune))
        batch = parameters_to_prune[start_idx:end_idx]

        print(
            f"  Pruning batch {batch_idx + 1}/{total_batches} ({len(batch)} layers)..."
        )

        # Apply pruning to this batch
        prune.global_unstructured(
            batch,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

        # Remove pruning hooks to make permanent
        for module, _ in batch:
            prune.remove(module, "weight")

        # Clear cache after each batch
        torch.cuda.empty_cache()

    print("L1 pruning complete.")
    return model


def save_glu_pruned_model(model, output_path, model_id):
    if hasattr(model.language_model, "config") and hasattr(
        model.language_model.config, "intermediate_size"
    ):
        model.config.intermediate_size = model.language_model.config.intermediate_size
        if (
            hasattr(model.config, "text_config")
            and model.config.text_config is not None
        ):
            model.config.text_config.intermediate_size = (
                model.language_model.config.intermediate_size
            )

    print(f"Saving GLU pruned model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_path)
    print(f"✓ Saved to {output_path}")


def save_head_pruned_model(model, output_path, model_id):
    print(f"Saving head pruned model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_path)
    print(f"✓ Saved to {output_path}")


def save_l1_pruned_model(model, output_path, model_id):
    print(f"Saving L1 pruned model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_path)
    print(f"✓ Saved to {output_path}")


def prune_glu(model_id, amount, output_path):
    print(f"\n{'=' * 60}")
    print(f"GLU Pruning - {amount * 100}%")
    print(f"{'=' * 60}")

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16
    ).to("cuda")

    glu_prune_model(model.language_model, prune_percent=amount)
    save_glu_pruned_model(model, output_path, model_id)

    del model
    torch.cuda.empty_cache()


def prune_heads(model_id, amount, output_path):
    print(f"\n{'=' * 60}")
    print(f"Attention Head Pruning - {amount * 100}%")
    print(f"{'=' * 60}")

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16
    ).to("cuda")

    prune_llama_heads_by_norm(model, amount)
    save_head_pruned_model(model, output_path, model_id)

    del model
    torch.cuda.empty_cache()


def prune_l1(model_id, amount, output_path):
    print(f"\n{'=' * 60}")
    print(f"L1 Unstructured Pruning - {amount * 100}%")
    print(f"{'=' * 60}")

    # Clear CUDA cache before loading
    torch.cuda.empty_cache()

    print("Loading model on GPU...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16
    ).to("cuda")

    apply_l1_pruning(model, amount)
    save_l1_pruned_model(model, output_path, model_id)

    del model
    torch.cuda.empty_cache()


def prune_combined_glu_heads(model_id, glu_amount, heads_amount, output_path):
    print(f"\n{'=' * 60}")
    print(f"Combined GLU ({glu_amount * 100}%) + Heads ({heads_amount * 100}%)")
    print(f"{'=' * 60}")

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16
    ).to("cuda")

    print("Step 1: GLU pruning...")
    glu_prune_model(model.language_model, prune_percent=glu_amount)

    print("Step 2: Head pruning...")
    prune_llama_heads_by_norm(model, heads_amount)

    save_glu_pruned_model(model, output_path, model_id)

    del model
    torch.cuda.empty_cache()


def main():
    model_id = "llava-hf/llava-1.5-7b-hf"
    os.makedirs("pruned_models", exist_ok=True)
    pruning_tasks = [
        # Single method pruning
        ("glu", 0.3, "llava-glu-30pct", prune_glu),
        ("glu", 0.7, "llava-glu-70pct", prune_glu),
        ("heads", 0.3, "llava-heads-30pct", prune_heads),
        ("heads", 0.7, "llava-heads-70pct", prune_heads),
        ("l1", 0.3, "llava-l1-30pct", prune_l1),
        ("l1", 0.7, "llava-l1-70pct", prune_l1),
    ]

    combined_tasks = [
        # Combined pruning
        (0.3, 0.3, "llava-glu30-heads30", prune_combined_glu_heads),
        (0.7, 0.7, "llava-glu70-heads70", prune_combined_glu_heads),
    ]

    print("\n" + "=" * 60)
    print("LLaVA Model Pruning Pipeline")
    print("=" * 60)
    print(f"Base model: {model_id}")
    print(f"Total tasks: {len(pruning_tasks) + len(combined_tasks)}")
    print("=" * 60 + "\n")

    for method, amount, output, func in pruning_tasks:
        try:
            output = os.path.join("pruned_models", output)
            func(model_id, amount, output)
        except Exception as e:
            print(f"✗ Failed: {e}")

    for amount1, amount2, output, func in combined_tasks:
        try:
            output = os.path.join("pruned_models", output)
            func(model_id, amount1, amount2, output)
        except Exception as e:
            print(f"✗ Failed: {e}")

    print("\n" + "=" * 60)
    print("All pruning tasks completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
