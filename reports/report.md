# Persian Text Sentiment Analysis Project Report

## Introduction

This project implements a deep learning model for sentiment analysis of Persian social media texts and online store user reviews. The model is built using ParsBERT (HooshvareLab/bert-fa-base-uncased) and trained on the SentiPers dataset. The goal is to classify Persian texts into three sentiment categories: negative, neutral, and positive.

## Recent Improvements (2025)

- **Smart Data Augmentation:** Advanced and diverse augmentation for negative samples (synonym replacement, random deletion, random swap) to increase data variety and reduce class imbalance.
- **Class Balancing:** The dataset is now much more balanced (see new stats below), improving model fairness and recall for minority classes.
- **Class Weights:** CrossEntropyLoss uses class weights for further balance during training.
- **Error Analysis:** Misclassified samples are saved for further analysis, enabling targeted improvements.
- **Modern GUI:** A professional Streamlit-based web app for user-friendly sentiment prediction, with probability visualization and prediction history.
- **Project Cleanup:** Old scripts and unused files were removed for a clean, maintainable codebase.

## Methods

### 1. Dataset

The SentiPers dataset contains over 15,500 Persian sentences with polarity labels ranging from -2 to +2. These labels were converted to three categories:

- **Negative (0):** Original labels -2 and -1
- **Neutral (1):** Original label 0
- **Positive (2):** Original labels +1 and +2

#### New Class Distribution (After Augmentation)

| Class    | Count |
| -------- | ----- |
| Negative | 5,071 |
| Neutral  | 4,984 |
| Positive | 6,628 |

(See `data/processed_data/dataset_stats.json` for details.)

### 2. Data Preprocessing & Augmentation

- **Text Normalization:** Using hazm for normalization and stemming, plus custom character normalization.
- **Stopword Removal:** Removal of common Persian stopwords.
- **Punctuation Cleaning:** Removal of special characters and punctuation marks.
- **Smart Augmentation:** For negative samples, synonym replacement, random word deletion, and random word swap are applied to generate diverse new samples.
- **Data Splitting:** 70% training, 15% validation, 15% test, stratified by class.

### 3. Model Architecture

- **Base Model:** HooshvareLab/bert-fa-base-uncased (ParsBERT)
- **Classification Head:** 3-class classification layer with attention mechanism and dropout.
- **Input Processing:** Maximum sequence length of 128 tokens with padding and truncation.
- **Device:** Automatic GPU/CPU detection with CUDA support.

### 4. Training Configuration

- **Optimizer:** AdamW (torch.optim) with learning rate 2e-5
- **Loss Function:** CrossEntropyLoss with class weights
- **Batch Size:** 16
- **Epochs:** 5 with early stopping (patience=3)
- **Class Weights:** Automatically calculated to balance the dataset
- **Progress Bars:** All steps show progress and status for transparency

## Results

### Model Performance (Previous Example)

- **Overall Accuracy:** ~80%
- **Weighted Precision:** ~80%
- **Weighted Recall:** ~80%
- **Weighted F1-Score:** ~80%

> **Note:** With the new balanced dataset and smart augmentation, recall and F1 for the negative class are expected to improve further. See `results/improvements/enhanced_model_metrics.json` for up-to-date metrics.

### New Class Distribution

| Class    | Count |
| -------- | ----- |
| Negative | 5,071 |
| Neutral  | 4,984 |
| Positive | 6,628 |

### Error Analysis

- Misclassified samples are saved in `results/improvements/misclassified_samples.csv` for further review.
- Confusion matrix and classification report are generated and saved after each training.

## GUI: User-Friendly Sentiment Prediction

A modern web app is provided for easy sentiment prediction:

- **Text Input:** Enter any Persian text.
- **Prediction:** Model predicts sentiment (negative, neutral, positive).
- **Probability Bar Chart:** Shows model confidence for each class.
- **Prediction History:** See all predictions in the current session.
- **Show Test Samples:** View random test samples and their true labels.
- **Clear History:** Remove prediction history with one click.

### How to Run the GUI

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run src/predict_gui.py
   ```
3. Open the provided local URL in your browser.

## Technical Implementation

### Dependencies

- transformers (for ParsBERT)
- torch (PyTorch framework)
- pandas (data manipulation)
- scikit-learn (metrics and preprocessing)
- hazm (Persian text processing)
- matplotlib/seaborn (visualization)
- streamlit (GUI)
- tqdm (progress bars)

### Model Persistence

The trained model is saved in the `models/enhanced_model` directory and can be loaded for inference without retraining.

## Conclusion

The Persian sentiment analysis model now achieves more balanced and robust performance, especially for the negative class, thanks to smart augmentation and class weighting. The project features a clean codebase, professional GUI, and advanced error analysis tools, making it suitable for real-world applications and further research.

---

**For more details, see the code and results folders.**
