
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
import matplotlib.lines as mlines
warnings.filterwarnings('ignore')
from matplotlib.lines import Line2D

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
CSV_PATH = "/userHome/userhome3/timur/vqa/benchmark_results.csv"
FIGURES_DIR = Path("/userHome/userhome3/timur/vqa/optimization_project/figures")

# Ensure figures directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Figure size and DPI for high-quality outputs
FIGSIZE_LARGE = (16, 10)
FIGSIZE_MEDIUM = (14, 8)
FIGSIZE_SMALL = (12, 6)
DPI = 300


def load_data():
    """Load and preprocess benchmark data."""
    df = pd.read_csv(CSV_PATH)
    
    # Extract base model name and pruning method
    df['base_model'] = df['model'].apply(lambda x: x.split(':')[0])
    df['pruning_variant'] = df['model'].apply(lambda x: x.split(':')[1] if ':' in x else 'original')
    
    # Parse pruning type and percentage
    df['pruning_type'] = df['pruning_variant'].apply(lambda x: 
        'glu' if 'glu' in x.lower() and 'heads' not in x.lower() else
        'heads' if 'heads' in x.lower() and 'glu' not in x.lower() else
        'l1' if 'l1' in x.lower() else
        'combined' if 'glu' in x.lower() and 'heads' in x.lower() else
        'none'
    )
    
    # Extract pruning percentage
    def extract_pct(name):
        if '30pct' in name or '30' in name:
            return 30
        elif '70pct' in name or '70' in name:
            return 70
        return 0
    
    df['pruning_pct'] = df['pruning_variant'].apply(extract_pct)
    
    # Add composite metrics
    df['quality_score'] = (df['meteor'] + df['rouge1'] + df['bertscore_f1']) / 3
    df['efficiency_score'] = df['quality_score'] / (df['avg_latency_s'] + 0.01)  # Add small epsilon
    df['size_reduction_pct'] = 100 * (1 - df.groupby('base_model')['model_size_mb'].transform(lambda x: x / x.max()))
    
    return df


def plot_efficiency_frontier_standard(df):
    """Plot efficiency frontier for standard models (excluding LLaVA pruned variants)."""
    
    # Filter for standard models only (original variants)
    standard_df = df[df['pruning_type'] == 'none'].copy()
    print(standard_df)
    
    markers = {'blip2': 'o', 'qwen': 's', 'paligemma': '^', 'llava': 'D'}
    colors = {'fp16': 'red', '8bit': 'blue', '4bit': 'green'}

    # Plot 1: Quality vs Latency
    plt.figure(figsize=FIGSIZE_LARGE)
    ax = plt.gca()
    
    for base_model in standard_df['base_model'].unique():
        for quant in standard_df['quantization'].unique():
            subset = standard_df[(standard_df['base_model'] == base_model) & (standard_df['quantization'] == quant)]
            if len(subset) > 0:
                ax.scatter(subset['avg_latency_s'], subset['quality_score'],
                          marker=markers.get(base_model, 'x'),
                          color=colors.get(quant, 'gray'),
                          s=150, alpha=0.7,
                          label=f'{base_model}-{quant}' if base_model == 'blip2' else '')
                
                # Add labels for standard models
                ax.annotate(f"{base_model}", 
                           (subset['avg_latency_s'].iloc[0], subset['quality_score'].iloc[0]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Average Latency (seconds)', fontsize=11)
    ax.set_ylabel('Quality Score', fontsize=11)
    ax.set_title('Standard Models: Quality vs Latency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='blip2', markerfacecolor='gray', markersize=10),
                      Line2D([0], [0], marker='s', color='w', label='qwen', markerfacecolor='gray', markersize=10),
                      Line2D([0], [0], marker='^', color='w', label='paligemma', markerfacecolor='gray', markersize=10),
                      Line2D([0], [0], marker='D', color='w', label='llava', markerfacecolor='gray', markersize=10),
                      Line2D([0], [0], marker='o', color='w', label='fp16', markerfacecolor='red', markersize=10),
                      Line2D([0], [0], marker='o', color='w', label='8bit', markerfacecolor='blue', markersize=10),
                      Line2D([0], [0], marker='o', color='w', label='4bit', markerfacecolor='green', markersize=10)]
    ax.legend(handles=legend_elements, loc='best', fontsize=9)
    
    output_path = FIGURES_DIR / "efficiency_frontier_standard_latency.png"
    plt.savefig(output_path, dpi=DPI)
    print(f"Saved {output_path}")
    plt.close()
    
    # Plot 2: Quality vs Size
    plt.figure(figsize=FIGSIZE_LARGE)
    ax = plt.gca()
    scatter_handles = {}

    for base_model in standard_df['base_model'].unique():
        subset = standard_df[standard_df['base_model'] == base_model]
        sc = ax.scatter(subset['model_size_mb'], subset['quality_score'],
                   s=1000 / (subset['avg_latency_s'] + 0.01),
                   alpha=0.6, label=base_model)
        scatter_handles[base_model] = sc.get_facecolor()[0]
        
        # Add labels
        for _, row in subset.iterrows():
            ax.annotate(f"{row['quantization']}", 
                       (row['model_size_mb'], row['quality_score']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    legend_handles = [
    mlines.Line2D([], [], marker='o', linestyle='None',
                  markersize=8, 
                  color=scatter_handles[base_model],
                  label=base_model)
    for base_model in scatter_handles
]

    ax.set_xlabel('Model Size (MB)', fontsize=11)
    ax.set_ylabel('Quality Score', fontsize=11)
    ax.set_title('Standard Models: Quality vs Size (bubble=speed)', fontsize=12, fontweight='bold')
    ax.legend(handles=legend_handles, fontsize=9)
    # ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    output_path = FIGURES_DIR / "efficiency_frontier_standard_size.png"
    plt.savefig(output_path, dpi=DPI)
    print(f"Saved {output_path}")
    plt.close()


def plot_efficiency_frontier_llama(df, dimension):
    """Plot efficiency frontier for standard models (excluding LLaVA pruned variants)."""
    colors = {'fp16': 'red', '8bit': 'blue', '4bit': 'green'}
    markers = {'glu': 'o', 'heads': 's', 'l1': '^', 'combined': 'D'}
    
    for quant in df['quantization'].unique():
        plt.figure(figsize=FIGSIZE_LARGE)
        ax = plt.gca()
        
        standard_df = df[df['pruning_type'] != 'none'].copy()
        standard_df = standard_df[standard_df['quantization'] == quant]


        for index, row in standard_df.iterrows():
            ax.scatter(row[dimension], row['quality_score'],
                            marker=markers.get(row['pruning_type'], 'x'),
                            color=colors.get(row['quantization'], 'gray'),
                            s=150, alpha=0.7)
            ax.annotate(row['model'].split(':')[1], 
                            (row[dimension], row['quality_score']),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
    
        ax.set_xlabel('Average Latency (seconds)' if dimension == 'avg_latency_s' else 'Model Size (MB)', fontsize=11)
        ax.set_ylabel('Quality Score', fontsize=11)
        ax.set_title(f'LLaVA Pruning: Quality vs {dimension} ({quant})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
        # Custom legend
        from matplotlib.lines import Line2D
        # legend_elements = []
        # for
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='glu', markerfacecolor='gray', markersize=10),
                        Line2D([0], [0], marker='s', color='w', label='heads', markerfacecolor='gray', markersize=10),
                        Line2D([0], [0], marker='^', color='w', label='l1', markerfacecolor='gray', markersize=10),
                        Line2D([0], [0], marker='D', color='w', label='combined', markerfacecolor='gray', markersize=10),]
        ax.legend(handles=legend_elements, loc='best', fontsize=9)
    
        output_path = FIGURES_DIR / f"efficiency_frontier_llama_{dimension}_{quant}.png"
        plt.savefig(output_path, dpi=DPI)
        print(f"Saved {output_path}")
        plt.close()


def plot_recommended_visualizations(df):
    """
    Creates:
    1. Two efficiency frontier scatter plots
    2. Bar plots (only fp16) with baseline dashed line:
        - Quality
        - Latency
        - Model size (optional)
    """

    df = df.copy()
    df["label"] = df["model"].apply(lambda x: x.split(":")[1] if ":" in x else x)

    quant_colors = {"fp16": "#D62728", "8bit": "#1F77B4", "4bit": "#2CA02C"}
    prune_markers = {"glu": "o", "heads": "s", "l1": "^", "combined": "D", "none": "x"}

    # ==========================================
    # 1. Scatter Plot: Latency vs Quality
    # ==========================================
    plt.figure(figsize=(10, 8))
    for _, row in df.iterrows():
        plt.scatter(row['avg_latency_s'], row['quality_score'],
                    s=row['model_size_mb'] * 0.1,
                    color=quant_colors.get(row['quantization'], "gray"),
                    marker=prune_markers.get(row['pruning_type'], "x"),
                    alpha=0.6)
        plt.text(row['avg_latency_s'], row['quality_score'], row['label'],
                 fontsize=8, alpha=0.7)

    plt.title("Efficiency Frontier: Latency vs Quality", fontsize=16, fontweight="bold")
    plt.xlabel("Latency (seconds)")
    plt.ylabel("Quality Score")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "efficiency_frontier_latency_vs_quality_all.png"
    plt.savefig(output_path, dpi=DPI)
    print(f"Saved {output_path}")
    plt.close()

    plt.figure(figsize=(10, 8))
    for _, row in df.iterrows():
        plt.scatter(row['model_size_mb'], row['quality_score'],
                    s=row['avg_latency_s'] * 150,
                    color=quant_colors.get(row['quantization'], "gray"),
                    marker=prune_markers.get(row['pruning_type'], "x"),
                    alpha=0.6)
        plt.text(row['model_size_mb'], row['quality_score'], row['label'],
                 fontsize=8, alpha=0.7)

    plt.title("Efficiency Frontier: Model size vs Quality", fontsize=16, fontweight="bold")
    plt.xlabel("Latency (seconds)")
    plt.ylabel("Quality Score")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "efficiency_frontier_model_size_vs_quality_all.png"
    plt.savefig(output_path, dpi=DPI)
    print(f"Saved {output_path}")
    plt.close()

    # ==========================================
    # 2. Scatter Plot: Model size vs Quality
    # ==========================================
    plt.figure(figsize=(10, 8))
    for _, row in df.iterrows():
        plt.scatter(row['model_size_mb'], row['quality_score'],
                    s=row['avg_latency_s'] * 120,
                    color=quant_colors.get(row['quantization'], "gray"),
                    marker=prune_markers.get(row['pruning_type'], "x"),
                    alpha=0.6)
        plt.text(row['model_size_mb'], row['quality_score'], row['label'],
                 fontsize=8, alpha=0.7)

    plt.title("Efficiency Frontier: Model Size vs Quality", fontsize=16, fontweight="bold")
    plt.xlabel("Model Size (MB)")
    plt.ylabel("Quality Score")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "efficiency_frontier_size_vs_quality_all.png"
    plt.savefig(output_path, dpi=DPI)
    print(f"Saved {output_path}")
    plt.close()

    # ==========================================
    # 3. BAR PLOTS (Only fp16) with baseline
    # ==========================================
    fp16 = df[df["quantization"] == "fp16"].copy()

    # Get baseline values (the non-pruned fp16 model)
    baseline_row = fp16[fp16["pruning_type"] == "none"].iloc[0]
    baseline_quality = baseline_row["quality_score"]
    baseline_latency = baseline_row["avg_latency_s"]
    baseline_size = baseline_row["model_size_mb"]

    # -------------------------
    # Bar: Quality
    # -------------------------
    plt.figure(figsize=(10, 6))
    plt.bar(fp16["label"], fp16["quality_score"], color="#D62728", alpha=0.85)
    plt.axhline(baseline_quality, linestyle="--", color="black", linewidth=2,
                label=f"Baseline ({baseline_row['label']})")

    plt.title("Quality Comparison (FP16 Only)", fontsize=14, fontweight="bold")
    plt.ylabel("Quality Score")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "comparison_fp16_quality.png"
    plt.savefig(output_path, dpi=DPI)
    print(f"Saved {output_path}")
    plt.close()

    # -------------------------
    # Bar: Latency
    # -------------------------
    plt.figure(figsize=(10, 6))
    plt.bar(fp16["label"], fp16["avg_latency_s"], color="#D62728", alpha=0.85)
    plt.axhline(baseline_latency, linestyle="--", color="black", linewidth=2,
                label=f"Baseline ({baseline_row['label']})")

    plt.title("Latency Comparison (FP16 Only)", fontsize=14, fontweight="bold")
    plt.ylabel("Average Latency (seconds)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "comparison_fp16_latency.png"
    plt.savefig(output_path, dpi=DPI)
    print(f"Saved {output_path}")
    plt.close()

    # -------------------------
    # Bar: Model Size (optional)
    # -------------------------
    plt.figure(figsize=(10, 6))
    plt.bar(fp16["label"], fp16["model_size_mb"], color="#D62728", alpha=0.85)
    plt.axhline(baseline_size, linestyle="--", color="black", linewidth=2,
                label=f"Baseline ({baseline_row['label']})")

    plt.title("Model Size Comparison (FP16 Only)", fontsize=14, fontweight="bold")
    plt.ylabel("Model Size (MB)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "comparison_fp16_size.png"
    plt.savefig(output_path, dpi=DPI)
    print(f"Saved {output_path}")
    plt.close()
    

if __name__ == "__main__":
    df = load_data()
    plot_efficiency_frontier_standard(df)
    llava_models = df[df['base_model'] == 'llava'][['model', 'quantization', 'pruning_type', 'pruning_pct', 'quality_score', 'efficiency_score', 'model_size_mb', 'size_reduction_pct', 'avg_latency_s']]
    plot_efficiency_frontier_llama(llava_models, 'avg_latency_s')
    plot_efficiency_frontier_llama(llava_models, 'model_size_mb')
    plot_recommended_visualizations(llava_models)
