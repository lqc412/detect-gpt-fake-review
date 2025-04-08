from typing import List, Dict
import json

def load_custom_dataset(file_path: str) -> List[str]:
    """Load custom dataset from a JSON file.
    
    Expected JSON format:
    {
        "human_texts": ["text1", "text2", ...],
        "gpt_texts": ["text1", "text2", ...]
    }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate data format
    assert "human_texts" in data and "gpt_texts" in data, \
        "JSON file must contain 'human_texts' and 'gpt_texts' keys"
    assert len(data["human_texts"]) == len(data["gpt_texts"]), \
        "Number of human and GPT texts must be equal"
    
    # Format data for DetectGPT
    formatted_data = {
        "original": data["human_texts"],
        "sampled": data["gpt_texts"]
    }
    
    return formatted_data

# Register custom dataset
DATASETS = {
    "custom": load_custom_dataset
}

def load(dataset_name: str, cache_dir: str, **kwargs) -> Dict[str, List[str]]:
    """Main loading function used by DetectGPT."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(DATASETS.keys())}")
    
    return DATASETS[dataset_name](**kwargs)