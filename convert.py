import pandas as pd
import json

def convert_csv_to_json(csv_path, output_path):
    """
    Convert CSV file to the required JSON format
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output JSON file
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Separate texts based on label
    human_texts = df[(df['label'] == 'OR') & (df['text_'].str.split().str.len() > 150) & (df['text_'].str.split().str.len() < 512)]['text_'].tolist()
    gpt_texts = df[(df['label'] == 'CG') & (df['text_'].str.split().str.len() > 150) & (df['text_'].str.split().str.len() < 512)]['text_'].tolist()
    
    minlen = min(len(human_texts), len(gpt_texts))
    human_texts = human_texts[:minlen]
    gpt_texts = gpt_texts[:minlen]

    # Create output dictionary
    output_data = {
        "human_texts": human_texts[:minlen],
        "gpt_texts": gpt_texts[:minlen]
    }
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"Processed {len(human_texts)} human texts and {len(gpt_texts)} GPT texts")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    input_path = "fake reviews dataset.csv"  # Change this to your CSV file path
    output_path = "complete_dataset.json"  # Change this to your desired output path
    
    convert_csv_to_json(input_path, output_path)