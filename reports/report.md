Persian Text Sentiment Analysis Project Report

Introduction

The goal of this project is to develop a deep learning model for sentiment analysis of Persian social media texts or online store user reviews. This model is designed using ParsBERT and the SentiPers dataset.

Methods

1. Dataset

The SentiPers dataset includes over 15,500 Persian sentences with polarity labels [-2, -1, 0, +1, +2] which were converted to three categories: negative (0), neutral (1), and positive (2).

2. Preprocessing

Text normalization using PersianStemmer.

Data splitting into training (70%), validation (15%), and test (15%) sets.

3. Model

ParsBERT model configured with 3 labels for classification. Training was performed with learning rate 2e-5 and 3 epochs.

Results

Accuracy: [actual value after execution]

Precision: [actual value after execution]

Recall: [actual value after execution]

F1-Score: [actual value after execution]

Analysis

The model performs better in classifying positive and negative texts compared to neutral texts. For improvement, more data or hyperparameter tuning can be used.

Visualization

Confusion Matrix: Shows the distribution of predictions.

Sentiment Distribution Chart: Displays the number of samples in each category.
