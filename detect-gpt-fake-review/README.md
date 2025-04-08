# 🛍️ Distinguishing Reality: Detecting AI-Generated Reviews in E-Commerce

This repository contains the source code, models, and report for our CS6140 final project. The goal is to build a robust detection system to classify human-written vs. AI-generated product reviews in e-commerce platforms using a mix of classical ML, deep learning, and zero-shot techniques.

---

## 📌 Project Highlights

- ✅ Achieved up to **98.77% accuracy** using BERT
- 🧠 Explored **DetectGPT** for **zero-shot** generalization
- 🔍 Performed **feature importance** and **error analysis**
- 📦 Includes dataset preprocessing, training scripts, and final paper

---

## 🧪 Methods Overview

| Model                  | Accuracy | Precision | Notes |
|------------------------|----------|-----------|-------|
| Baseline Classifier    | 86.65%   | 87.78%    | TF-IDF + DecisionTree |
| Decision Tree          | 76.90%   | 77.52%    | Overfitting issue |
| Random Forest          | 88.47%   | 89.00%    | Good generalization |
| BERT (Fine-tuned)      | 98.77%   | 98.65%    | Best performance |
| DetectGPT (Zero-shot)  | ROC-AUC: 0.979 | – | Effective on unseen data |

---

## 📂 Project Structure

```
.
├── CS6140_Fake_Reviews_Detection.ipynb    # Main Jupyter notebook
├── detect_custom.py                       # Custom DetectGPT implementation
├── convert.py                             # Feature engineering logic
├── my_custom_datasets.py                  # Dataset loader for BERT and classical ML
├── dataset.json                           # Input dataset (labeled reviews)
├── Detecting_AI-Generated_Reviews_in_E-Commerce.pdf  # Final formatted report
└── README.md                              # This file
```

---

## 📊 Features Used

- **Textual**: TF-IDF (3000), Sentence count, Word count, Char count
- **Categorical**: One-hot encoding for product categories
- **Rating**: Normalized numeric rating
- **Deep Features**: BERT embeddings (Huggingface `bert-base-uncased`)
- **Zero-shot Score**: Log probability curvature from DetectGPT

---

## 📚 Dataset

We used the [Fake Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset/data), which includes:
- `CG` (ChatGPT-generated reviews)
- `OR` (original human-written reviews)

All data were preprocessed using `convert.py` and split for training/testing.

---

## 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve & AUC
- Feature Importance Visualization


---

## 🛠️ Built With

- Python, scikit-learn, PyTorch, Transformers
- BERT (Huggingface)
- DetectGPT (custom implementation)
- Jupyter, Matplotlib, Seaborn

---
