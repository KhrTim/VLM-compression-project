from huggingface_hub import snapshot_download

# Download from a dataset
snapshot_download(
    repo_id="ChongyanChen/VQAonline",
    repo_type="dataset",
    max_workers=64,
    token=True,
    cache_dir="./cache",
    local_dir="./data",
)
