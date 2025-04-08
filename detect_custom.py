import argparse
import torch
import transformers
import numpy as np
import time
import my_custom_datasets
import functools
import tqdm
import re
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

cache_dir = ".cache"
DEVICE = "cuda"
n_perturbation_rounds = 1
pattern = re.compile(r"<extra_id_\d+>")

def load_mask_model_and_tokenizer(model_name: str, device: str):
    """Load mask filling model and tokenizer"""
    print(f'Loading mask filling model {model_name}...')
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=cache_dir
    ).to(device)
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=512,
        cache_dir=cache_dir
    )
    return mask_model, mask_tokenizer

def load_mask_model(base_model, mask_model):
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()

    base_model.cpu()
    mask_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')

def load_base_model(base_model, mask_model):
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()
    try:
        mask_model.cpu()
    except NameError:
        pass
    base_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')

def perturb_texts(args, mask_model, mask_tokenizer, texts, span_length, pct, ceil_pct=False):
    chunk_size = args.chunk_size
    if '11b' in args.mask_filling_model_name:
        chunk_size //= 2

    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(args, mask_model, mask_tokenizer, texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs

def tokenize_and_mask(args, mask_tokenizer, text, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text

def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

def replace_masks(texts, mask_model, mask_tokenizer):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=1.0, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts

def perturb_texts_(args, mask_model, mask_tokenizer, texts, span_length, pct, ceil_pct=False):
    masked_texts = [tokenize_and_mask(args, mask_tokenizer, x, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(masked_texts, mask_model, mask_tokenizer)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(args, mask_tokenizer, x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(masked_texts, mask_model, mask_tokenizer)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1
    return perturbed_texts

def print_classification_results(outputs, data):
    """Print classification results and scores for each text"""
    for output in outputs:
        print(f"\n=== {output['name']} Classification Results ===")
        
        predictions = output['predictions']
        real_scores = predictions['real']  # scores for original texts
        sample_scores = predictions['samples']  # scores for sampled texts
        
        print("\nResults for Human-written Texts:")
        for i, (text, score) in enumerate(zip(data['original'], real_scores)):
            print(f"\nText {i+1}:")
            print(f"Content: {text[:100]}..." if len(text) > 100 else text)
            print(f"Score: {score:.4f}")
        
        print("\nResults for AI-generated Texts:")
        for i, (text, score) in enumerate(zip(data['sampled'], sample_scores)):
            print(f"\nText {i+1}:")
            print(f"Content: {text[:100]}..." if len(text) > 100 else text)
            print(f"Score: {score:.4f}")
            
        # Calculate average scores
        print(f"\nAverage Scores:")
        print(f"Human-written texts: {sum(real_scores)/len(real_scores):.4f}")
        print(f"AI-generated texts: {sum(sample_scores)/len(sample_scores):.4f}")

def load_base_model_and_tokenizer(args, name):
    print(f'Loading BASE model {args.base_model_name}...')
    base_model_kwargs = {}
    base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=cache_dir)

    optional_tok_kwargs = {}
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer

# Get the log likelihood of each text under the base_model
def get_ll(text, base_model, base_tokenizer):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()


def get_lls(texts, base_model, base_tokenizer):
    return [get_ll(text, base_model, base_tokenizer) for text in texts]

def get_perturbation_results(args, base_model, base_tokenizer, mask_model, mask_tokenizer, data, span_length=10, n_perturbations=1, n_samples=500):
    load_mask_model(base_model, mask_model)

    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    original_text = data["original"]
    sampled_text = data["sampled"]

    perturb_fn = functools.partial(perturb_texts, args, mask_model, mask_tokenizer, span_length=span_length, pct=args.pct_words_masked)

    p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
    p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
    for _ in range(n_perturbation_rounds - 1):
        try:
            p_sampled_text, p_original_text = perturb_fn(p_sampled_text), perturb_fn(p_original_text)
        except AssertionError:
            break

    assert len(p_sampled_text) == len(sampled_text) * n_perturbations, f"Expected {len(sampled_text) * n_perturbations} perturbed samples, got {len(p_sampled_text)}"
    assert len(p_original_text) == len(original_text) * n_perturbations, f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

    for idx in range(len(original_text)):
        results.append({
            "original": original_text[idx],
            "sampled": sampled_text[idx],
            "perturbed_sampled": p_sampled_text[idx * n_perturbations: (idx + 1) * n_perturbations],
            "perturbed_original": p_original_text[idx * n_perturbations: (idx + 1) * n_perturbations]
        })

    load_base_model(base_model, mask_model)

    for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
        p_sampled_ll = get_lls(res["perturbed_sampled"], base_model, base_tokenizer)
        p_original_ll = get_lls(res["perturbed_original"], base_model, base_tokenizer)
        res["original_ll"] = get_ll(res["original"], base_model, base_tokenizer)
        res["sampled_ll"] = get_ll(res["sampled"], base_model, base_tokenizer)
        res["all_perturbed_sampled_ll"] = p_sampled_ll
        res["all_perturbed_original_ll"] = p_original_ll
        res["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
        res["perturbed_original_ll"] = np.mean(p_original_ll)
        res["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
        res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

    return results

def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

def run_perturbation_experiment(args, results, criterion, span_length=10, n_perturbations=1, n_samples=500):
    # compute diffs with perturbed
    predictions = {'real': [], 'samples': []}
    for res in results:
        if criterion == 'd':
            predictions['real'].append(res['original_ll'] - res['perturbed_original_ll'])
            predictions['samples'].append(res['sampled_ll'] - res['perturbed_sampled_ll'])
        elif criterion == 'z':
            if res['perturbed_original_ll_std'] == 0:
                res['perturbed_original_ll_std'] = 1
                print("WARNING: std of perturbed original is 0, setting to 1")
                print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
                print(f"Original text: {res['original']}")
            if res['perturbed_sampled_ll_std'] == 0:
                res['perturbed_sampled_ll_std'] = 1
                print("WARNING: std of perturbed sampled is 0, setting to 1")
                print(f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}")
                print(f"Sampled text: {res['sampled']}")
            predictions['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
            predictions['samples'].append((res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    name = f'perturbation_{n_perturbations}_{criterion}'
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': name,
        'predictions': predictions,
        'info': {
            'pct_words_masked': args.pct_words_masked,
            'span_length': span_length,
            'n_perturbations': n_perturbations,
            'n_samples': n_samples,
        },
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }

def save_roc_curves(args, base_model_name, experiments):
    # first, clear plt
    plt.clf()

    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["metrics"]
        plt.plot(metrics["fpr"], metrics["tpr"], label=f"{experiment['name']}, roc_auc={metrics['roc_auc']:.3f}", color=color)
        # print roc_auc for this experiment
        print(f"{experiment['name']} roc_auc: {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves ({base_model_name} - {args.mask_filling_model_name})')
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(f"custom_result/roc_curves.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to JSON file containing custom dataset')
    parser.add_argument('--base_model_name', type=str, default="gpt2",
                       help='Name of the base model to use for detection')
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-large",
                       help='Name of the mask filling model')
    parser.add_argument('--pct_words_masked', type=float, default=0.3,
                       help='Percentage of words to mask')
    parser.add_argument('--span_length', type=int, default=2,
                       help='Length of spans to mask')
    parser.add_argument('--n_perturbations', type=int, default=10,
                       help='Number of perturbations per text')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--cache_dir', type=str, default=".cache",
                       help='Cache directory for models')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--chunk_size', type=int, default=20)
    args = parser.parse_args()

    
    # Load base model and tokenizer
    print(f'Loading base model {args.base_model_name}...')
    base_model, base_tokenizer = load_base_model_and_tokenizer(args, args.base_model_name)
    base_model = base_model.to(DEVICE)
    
    # Load mask filling model and tokenizer
    mask_model, mask_tokenizer = load_mask_model_and_tokenizer(args.mask_filling_model_name, DEVICE)
    
    # Make models and tokenizers available globally as expected by original code
    globals().update({
        'base_model': base_model,
        'base_tokenizer': base_tokenizer,
        'mask_model': mask_model,
        'mask_tokenizer': mask_tokenizer,
        'DEVICE': DEVICE,
        'args': args
    })
    
    # Load custom dataset
    data = my_custom_datasets.load("custom", args.cache_dir, file_path=args.data_path)
    globals()['data'] = data
    
    # Run detection
    perturbation_results = get_perturbation_results(
        args,
        base_model, base_tokenizer, mask_model, mask_tokenizer, data,
        span_length=args.span_length,
        n_perturbations=args.n_perturbations,
        n_samples=len(data["original"])
    )
    
    # Get results for both metrics
    outputs = []
    for perturbation_mode in ['d', 'z']:
        output = run_perturbation_experiment(
            args,
            perturbation_results,
            perturbation_mode,
            span_length=args.span_length,
            n_perturbations=args.n_perturbations,
            n_samples=len(data["original"])
        )
        outputs.append(output)
    print_classification_results(outputs, data)
        
    # Save results
    save_roc_curves(args, args.base_model_name, outputs)
    
    # Print results
    for output in outputs:
        print(f"\nResults for {output['name']}:")
        print(f"ROC AUC: {output['metrics']['roc_auc']:.3f}")
        print(f"PR AUC: {output['pr_metrics']['pr_auc']:.3f}")

if __name__ == "__main__":
    main()