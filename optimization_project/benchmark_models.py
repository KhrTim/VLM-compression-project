import argparse
import time
import torch
import evaluate
from PIL import Image
import os
import sys
import gc
import pandas as pd
import json
import shutil

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_train_dataset
from vision_language_models import load_model

MODELS = [
    "blip2",
    "qwen",
    "paligemma",
    "llava",
    "llava:llava-glu-30pct",
    "llava:llava-glu-70pct",
    "llava:llava-heads-30pct",
    "llava:llava-heads-70pct",
    "llava:llava-l1-30pct",
    "llava:llava-l1-70pct",
    "llava:llava-glu30-heads30",
    "llava:llava-glu70-heads70",
]

def clean_text(text):
    """Clean and normalize text by removing newlines and extra whitespace."""
    if not text:
        return ""
    # Replace newlines with spaces
    text = text.replace("\n", " ").replace("\r", " ")
    # Normalize multiple spaces to single space
    text = " ".join(text.split())
    return text.strip()


def save_detailed_results(
    model_name, quantization, dataset, ground_truths, predictions
):
    # Save detailed results to JSON (text already cleaned)
    json_output_file = os.path.join("tmp", f"{model_name}_{quantization}_results.json")
    results_data = []
    for i in range(len(predictions)):
        results_data.append(
            {
                "question": dataset[i]["question"],
                "ground_truth": ground_truths[i],
                "prediction": predictions[i],
            }
        )

    with open(json_output_file, "w") as f:
        json.dump(results_data, f, indent=4)
    print(f"Saved detailed results to {json_output_file}")


def evaluate_model_on_sample(ground_truth_answers, generated_answers):
    # Text is already cleaned when appended to lists
    # Filter out empty predictions to avoid tokenization errors
    # Replace empty strings with a placeholder
    filtered_predictions = [
        pred if pred.strip() else "[NO ANSWER]" for pred in generated_answers
    ]
    filtered_references = [
        ref if ref.strip() else "[NO ANSWER]" for ref in ground_truth_answers
    ]

    # Combine METEOR and SacreBLEU
    clf_metrics = evaluate.combine(
        [
            "evaluate-metric/meteor",
            "evaluate-metric/sacrebleu",
        ]
    )

    # Load additional metrics
    perplexity_metric = evaluate.load("perplexity", module_type="metric")
    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")

    # Compute combined metrics
    for ref, pred in zip(filtered_references, filtered_predictions):
        clf_metrics.add(references=ref, predictions=pred)
    meteor_and_sacrebleu_metrics = clf_metrics.compute()

    # Compute perplexity (only on non-empty predictions)
    perplexity_score = perplexity_metric.compute(
        predictions=filtered_predictions, model_id="gpt2"
    )

    # Compute ROUGE
    rouge_score = rouge_metric.compute(
        predictions=filtered_predictions, references=filtered_references
    )

    # Compute BERTScore
    bertscore_results = bertscore_metric.compute(
        predictions=filtered_predictions, references=filtered_references, lang="en"
    )

    # Calculate mean answer length
    mean_answer_length = (
        sum(len(pred) for pred in generated_answers) / len(generated_answers)
        if generated_answers
        else 0
    )

    # Merge all metrics into one dictionary
    all_metrics = {**meteor_and_sacrebleu_metrics}
    all_metrics["rouge1"] = rouge_score["rouge1"]
    all_metrics["rouge2"] = rouge_score["rouge2"]
    all_metrics["rougeL"] = rouge_score["rougeL"]
    all_metrics["bertscore_precision"] = sum(bertscore_results["precision"]) / len(
        bertscore_results["precision"]
    )
    all_metrics["bertscore_recall"] = sum(bertscore_results["recall"]) / len(
        bertscore_results["recall"]
    )
    all_metrics["bertscore_f1"] = sum(bertscore_results["f1"]) / len(
        bertscore_results["f1"]
    )
    all_metrics["mean_answer_length"] = mean_answer_length

    return all_metrics, perplexity_score


def benchmark(model_name, quantization, dataset, num_samples=10):
    print(f"Benchmarking {model_name} with {quantization} quantization...")

    # Measure Memory (Before Load)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_load = time.time()
    model = load_model(model_name, quantization=quantization)
    load_time = time.time() - start_load
    memory_usage_bytes = model.calculate_model_size_in_bytes()
    model_parameters_count = model.calculate_model_parameters()

    latencies = []
    ground_truths = []
    predictions = []

    for i in range(num_samples):
        sample = dataset[i]
        image = Image.open(sample["image"]).convert("RGB")
        question = sample["question"]
        answer = sample["answer"]

        start_gen = time.time()
        pred = model.generate(image, question)
        latencies.append(time.time() - start_gen)

        # Clean text once when appending (removes newlines and normalizes whitespace)
        ground_truths.append(clean_text(answer))
        predictions.append(clean_text(pred))
        print(
            f"Sample {i + 1}/{num_samples}: GT='{answer[:100]}', Pred='{pred[:100]}', Time={latencies[-1]:.2f}s\n"
        )

    avg_latency = sum(latencies) / len(latencies)
    print(f"Average Latency: {avg_latency:.2f}s")

    save_detailed_results(model_name, quantization, dataset, ground_truths, predictions)

    meteor_and_sacrebleu_metrics, perplexity_score = evaluate_model_on_sample(
        ground_truths, predictions
    )
    print(f"Accuracy Scores: {meteor_and_sacrebleu_metrics}")
    print(f"Perplexity Score: {perplexity_score}")

    model.cleanup()

    return {
        "model": model_name,
        "quantization": quantization,
        "load_time": load_time,
        "avg_latency": avg_latency,
        "perplexity_score": perplexity_score,
        "model_size_bytes": memory_usage_bytes,
        "model_parameters": model_parameters_count,
        "scores": meteor_and_sacrebleu_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=MODELS + ["all"])
    parser.add_argument(
        "--quantization",
        type=str,
        default="fp16",
        choices=["fp16", "4bit", "8bit", "all"],
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output_file", type=str, default="benchmark_results.csv")
    args = parser.parse_args()

    models_to_run = MODELS if args.model == "all" else [args.model]
    quantizations_to_run = (
        ["fp16", "8bit", "4bit"] if args.quantization == "all" else [args.quantization]
    )

    # Cleanup and recreate tmp directory
    tmp_dir = "tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    all_results = []
    dataset = load_train_dataset()

    for model_name in models_to_run:
        for quant in quantizations_to_run:
            try:
                print(f"\n=== Running Benchmark: {model_name} | {quant} ===")
                result = benchmark(model_name, quant, dataset, args.samples)

                # Flatten scores for CSV
                flat_result = {
                    "model": result["model"],
                    "quantization": result["quantization"],
                    "load_time_s": result["load_time"],
                    "model_size_mb": result["model_size_bytes"] / (1024 * 1024),
                    "model_parameters": result["model_parameters"],
                    "avg_latency_s": result["avg_latency"],
                    "mean_answer_length": result["scores"]["mean_answer_length"],
                    "meteor": result["scores"]["meteor"],
                    "sacrebleu": result["scores"]["score"],
                    "rouge1": result["scores"]["rouge1"],
                    "rouge2": result["scores"]["rouge2"],
                    "rougeL": result["scores"]["rougeL"],
                    "bertscore_precision": result["scores"]["bertscore_precision"],
                    "bertscore_recall": result["scores"]["bertscore_recall"],
                    "bertscore_f1": result["scores"]["bertscore_f1"],
                    "perplexity": result["perplexity_score"]["mean_perplexity"],
                }
                all_results.append(flat_result)

                # Clean up to avoid OOM
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Failed to benchmark {model_name} with {quant}: {e}")

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(args.output_file, index=False)
        print(f"\nBenchmark complete. Results saved to {args.output_file}")
        print(df)
    else:
        print("No benchmarks completed successfully.")
