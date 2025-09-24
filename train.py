"""Command line interface for training fake review detectors."""
from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

LOGGER = logging.getLogger("train")

DEFAULT_RANDOM_STATE = 42
SUPPORTED_CLASSICAL_MODELS = {
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
}
SUPPORTED_TRANSFORMERS = {
    "bert": "bert-base-uncased",
    "gpt2": "gpt2",
}
SUPPORTED_FEATURES = {"tfidf", "onehot", "counts", "rating"}


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fake review detectors with cross validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(set(SUPPORTED_CLASSICAL_MODELS) | set(SUPPORTED_TRANSFORMERS)),
        help="Model family to train.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified folds.",
    )
    parser.add_argument(
        "--features",
        default="tfidf,onehot,counts,rating",
        help=(
            "Comma separated list of feature groups for classical models. "
            "Supported groups: tfidf, onehot, counts, rating. Use 'all' for every feature."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where checkpoints and metrics will be saved.",
    )
    parser.add_argument(
        "--data",
        default="fake reviews dataset.csv",
        help="Training data file (CSV with labels or JSON produced by convert.py).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for cross validation.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for transformer models.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Training epochs for transformer models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for transformer fine-tuning.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for transformer fine-tuning.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable dataset caching used by the transformers Dataset.map pipeline.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def load_reviews(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find training data at {data_path}")

    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
        expected_columns = {"text_", "label"}
        if not expected_columns.issubset(df.columns):
            raise ValueError(
                f"CSV file must contain the columns {sorted(expected_columns)}. Found {sorted(df.columns)}"
            )
        df = df.copy()
        if "category" not in df.columns:
            df["category"] = "unknown"
        if "rating" not in df.columns:
            df["rating"] = np.nan
    elif data_path.suffix.lower() == ".json":
        with data_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if "human_texts" not in payload or "gpt_texts" not in payload:
            raise ValueError("JSON file must contain 'human_texts' and 'gpt_texts' arrays")
        human = list(payload["human_texts"])
        gpt = list(payload["gpt_texts"])
        texts = human + gpt
        labels = ["OR"] * len(human) + ["CG"] * len(gpt)
        df = pd.DataFrame(
            {
                "text_": texts,
                "label": labels,
                "category": "unknown",
                "rating": np.nan,
            }
        )
    else:
        raise ValueError(f"Unsupported file extension for {data_path}")

    df["text_"] = df["text_"].astype(str).fillna("")
    df["label"] = df["label"].astype(str)
    df["category"] = df.get("category", "unknown").astype(str).fillna("unknown")
    df["rating"] = pd.to_numeric(df.get("rating", np.nan), errors="coerce")

    label_map = {"CG": 0, "OR": 1, "0": 0, "1": 1, 0: 0, 1: 1}
    if not set(df["label"]).issubset(label_map):
        raise ValueError("Labels must be 'CG' and 'OR' for generated and original reviews respectively")

    df["target"] = df["label"].map(label_map).astype(int)
    return df


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_sentences(text: str) -> int:
    pieces = re.split(r"[.!?]+", text)
    return sum(1 for piece in pieces if piece.strip())


def build_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric = pd.DataFrame(index=df.index)
    numeric["num_sentences"] = df["text_"].apply(count_sentences).astype(float)
    numeric["num_words"] = df["norm_text_"].apply(lambda x: len(x.split()))
    numeric["num_characters"] = df["norm_text_"].apply(len)
    rating_series = df["rating"].astype(float)
    median_rating = rating_series.median()
    if np.isnan(median_rating):
        median_rating = 0.0
    numeric["rating"] = rating_series.fillna(median_rating)
    return numeric


def _build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)


def parse_feature_set(raw: str) -> List[str]:
    if not raw:
        return []
    raw = raw.strip()
    if raw.lower() == "all":
        return sorted(SUPPORTED_FEATURES)
    requested = {feature.strip().lower() for feature in raw.split(",") if feature.strip()}
    invalid = requested - SUPPORTED_FEATURES
    if invalid:
        raise ValueError(f"Unsupported features requested: {sorted(invalid)}")
    return sorted(requested)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def run_classical_training(args: argparse.Namespace, df: pd.DataFrame) -> List[Dict[str, float]]:
    feature_groups = parse_feature_set(args.features)
    if not feature_groups:
        raise ValueError("At least one feature group must be provided for classical models")

    df = df.copy()
    df["norm_text_"] = df["text_"].apply(preprocess_text)

    numeric_features = build_numeric_features(df)

    categories = df["category"].fillna("unknown").astype(str).values.reshape(-1, 1)
    texts = df["norm_text_"].values
    y = df["target"].values

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    per_fold_metrics: List[Dict[str, float]] = []
    model_cls = SUPPORTED_CLASSICAL_MODELS[args.model]

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, y), start=1):
        LOGGER.info("Starting fold %s/%s", fold_idx, args.folds)
        fold_dir = Path(args.output_dir) / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        vectorizer = None
        encoder = None
        train_metadata = {}

        if "tfidf" in feature_groups:
            vectorizer = TfidfVectorizer(max_features=3000)
            vectorizer.fit(texts[train_idx])
            train_metadata["tfidf"] = {
                "max_features": vectorizer.max_features,
                "vocabulary_size": len(vectorizer.vocabulary_),
            }
            train_tfidf = vectorizer.transform(texts[train_idx])
            val_tfidf = vectorizer.transform(texts[val_idx])
        else:
            train_tfidf = None
            val_tfidf = None

        if "onehot" in feature_groups:
            encoder = _build_one_hot_encoder()
            encoder.fit(categories[train_idx])
            train_metadata["onehot"] = {
                "categories": [list(cat) for cat in encoder.categories_],
            }
            train_onehot = encoder.transform(categories[train_idx])
            val_onehot = encoder.transform(categories[val_idx])
        else:
            train_onehot = None
            val_onehot = None

        numeric_part_names: List[str] = []
        numeric_train_parts: List[csr_matrix] = []
        numeric_val_parts: List[csr_matrix] = []

        if "counts" in feature_groups:
            numeric_part_names.extend(["num_sentences", "num_words", "num_characters"])
        if "rating" in feature_groups:
            numeric_part_names.append("rating")

        if numeric_part_names:
            numeric_matrix = csr_matrix(numeric_features.loc[:, numeric_part_names].values.astype(float))
            numeric_train_parts.append(numeric_matrix[train_idx])
            numeric_val_parts.append(numeric_matrix[val_idx])
            train_metadata["numeric_features"] = numeric_part_names

        train_matrices = []
        val_matrices = []
        if train_tfidf is not None:
            train_matrices.append(train_tfidf)
            val_matrices.append(val_tfidf)
        if train_onehot is not None:
            train_matrices.append(train_onehot)
            val_matrices.append(val_onehot)
        if numeric_train_parts:
            train_numeric = hstack(numeric_train_parts) if len(numeric_train_parts) > 1 else numeric_train_parts[0]
            val_numeric = hstack(numeric_val_parts) if len(numeric_val_parts) > 1 else numeric_val_parts[0]
            train_matrices.append(train_numeric)
            val_matrices.append(val_numeric)

        if not train_matrices:
            raise ValueError("No features were assembled for training")

        X_train = hstack(train_matrices).tocsr()
        X_val = hstack(val_matrices).tocsr()

        model_kwargs = {"random_state": args.seed}
        if args.model == "random_forest":
            model_kwargs.update({"n_estimators": 200, "n_jobs": -1})

        model = model_cls(**model_kwargs)
        model.fit(X_train, y[train_idx])

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = compute_binary_metrics(y[val_idx], y_pred, y_prob)
        metrics["fold"] = fold_idx
        metrics["support"] = int(len(val_idx))
        per_fold_metrics.append(metrics)

        LOGGER.info(
            "Fold %s metrics: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%s",
            fold_idx,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            "{:.4f}".format(metrics["roc_auc"]) if not math.isnan(metrics["roc_auc"]) else "nan",
        )

        report = classification_report(
            y[val_idx],
            y_pred,
            target_names=["generated", "original"],
            zero_division=0,
            output_dict=True,
        )
        save_json(report, fold_dir / "classification_report.json")

        with (fold_dir / "model.pkl").open("wb") as fp:
            pickle.dump(model, fp)
        if vectorizer is not None:
            with (fold_dir / "vectorizer.pkl").open("wb") as fp:
                pickle.dump(vectorizer, fp)
        if encoder is not None:
            with (fold_dir / "encoder.pkl").open("wb") as fp:
                pickle.dump(encoder, fp)

        save_json(
            {
                "features": feature_groups,
                "metadata": train_metadata,
            },
            fold_dir / "feature_config.json",
        )

    metrics_df = pd.DataFrame(per_fold_metrics)
    metrics_df.to_csv(Path(args.output_dir) / "metrics.csv", index=False)

    aggregate = {
        metric: {
            "mean": float(metrics_df[metric].mean()),
            "std": float(metrics_df[metric].std(ddof=0)),
        }
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    }

    save_json(
        {
            "per_fold": per_fold_metrics,
            "aggregate": aggregate,
            "folds": args.folds,
            "model": args.model,
            "features": feature_groups,
        },
        Path(args.output_dir) / "metrics.json",
    )

    return per_fold_metrics


def run_transformer_training(args: argparse.Namespace, df: pd.DataFrame) -> List[Dict[str, float]]:
    try:
        from datasets import Dataset
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  Trainer, TrainingArguments, set_seed)
    except ImportError as exc:
        raise RuntimeError("Transformer training requires the 'datasets' and 'transformers' packages") from exc

    model_name = args.model
    if model_name not in SUPPORTED_TRANSFORMERS:
        raise ValueError(f"Unsupported transformer model '{model_name}'")
    pretrained_name = SUPPORTED_TRANSFORMERS[model_name]

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    if model_name == "gpt2" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    df = df.copy()
    df["text"] = df["text_"].astype(str)
    df["label"] = df["target"].astype(int)

    dataset = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, load_from_cache_file=not args.no_cache)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    y = df["label"].to_numpy()
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    per_fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        LOGGER.info("Starting transformer fold %s/%s", fold_idx, args.folds)
        set_seed(args.seed + fold_idx)

        train_dataset = tokenized.select(train_idx.tolist())
        eval_dataset = tokenized.select(val_idx.tolist())

        fold_dir = Path(args.output_dir) / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_name,
            num_labels=2,
        )
        if model_name == "gpt2":
            model.resize_token_embeddings(len(tokenizer))

        training_args = TrainingArguments(
            output_dir=str(fold_dir / "checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            evaluation_strategy="no",
            save_strategy="no",
            logging_dir=str(fold_dir / "logs"),
            logging_steps=50,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()

        predictions = trainer.predict(eval_dataset)
        logits = predictions.predictions
        probs = _softmax(logits)
        y_pred = probs.argmax(axis=1)
        y_prob = probs[:, 1]

        metrics = compute_binary_metrics(y[val_idx], y_pred, y_prob)
        metrics["fold"] = fold_idx
        metrics["support"] = int(len(val_idx))
        per_fold_metrics.append(metrics)

        LOGGER.info(
            "Transformer fold %s metrics: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
            fold_idx,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
        )

        report = classification_report(
            y[val_idx],
            y_pred,
            target_names=["generated", "original"],
            zero_division=0,
            output_dict=True,
        )
        save_json(report, fold_dir / "classification_report.json")

        trainer.save_model(str(fold_dir / "model"))
        tokenizer.save_pretrained(str(fold_dir / "tokenizer"))

        save_json(
            {
                "model_name": pretrained_name,
                "max_length": args.max_length,
            },
            fold_dir / "model_config.json",
        )

    metrics_df = pd.DataFrame(per_fold_metrics)
    metrics_df.to_csv(Path(args.output_dir) / "metrics.csv", index=False)

    aggregate = {
        metric: {
            "mean": float(metrics_df[metric].mean()),
            "std": float(metrics_df[metric].std(ddof=0)),
        }
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    }

    save_json(
        {
            "per_fold": per_fold_metrics,
            "aggregate": aggregate,
            "folds": args.folds,
            "model": args.model,
            "pretrained_model": pretrained_name,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        Path(args.output_dir) / "metrics.json",
    )

    return per_fold_metrics


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 1:
        logits = np.stack([-logits, logits], axis=1)
    logits = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(logits)
    sums = exps.sum(axis=1, keepdims=True)
    return exps / sums


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_reviews(Path(args.data))
    LOGGER.info("Loaded %s reviews from %s", len(df), args.data)

    if args.model in SUPPORTED_CLASSICAL_MODELS:
        run_classical_training(args, df)
    else:
        run_transformer_training(args, df)


if __name__ == "__main__":
    main()
