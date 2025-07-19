import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.visualize_results import visualize_results
from src.evaluate_model import evaluate_model
from src.train_model import train_model
from src.dataset import SentimentDataset, DataLoader
from src.setup_model import tokenizer, model, device
from src.preprocess_data import load_and_preprocess_data
from src.data_augmentation import balance_dataset
import os
import sys
sys.path.append('src')


def predict_examples(model, tokenizer, examples, device):
    model.eval()
    results = []
    with torch.no_grad():
        for text in examples:
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=128,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy().tolist()[0]
            pred = int(torch.argmax(outputs.logits, dim=1).cpu().item())
            results.append((text, logits, pred))
    # Save log
    with open('reports/sample_predictions.log', 'w', encoding='utf-8') as f:
        for text, logits, pred in results:
            f.write(
                f"Text: {text}\nLogits: {logits}\nPredicted label: {pred}\n\n")
    return results


def main():
    model_path = 'models/parsbert_sentiment_model'

    # Check if trained model exists
    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, 'config.json')):
        print("Trained model found. Loading...")

        # Load model and tokenizer from saved files
        loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
        loaded_model = AutoModelForSequenceClassification.from_pretrained(
            model_path)
        loaded_model.to(device)

        print("Model loaded successfully!")

        # Perform sentiment analysis on sample texts
        sample_texts = [
            "این فیلم خیلی خوب بود و من لذت بردم.",
            "کیفیت محصول اصلا خوب نبود و ناراضی‌ام.",
            "قیمتش معمولی بود و تفاوتی با بقیه نداشت.",
            "خدمات عالی بود و حتما دوباره استفاده می‌کنم.",
            "تجربه بدی داشتم و پیشنهاد نمی‌کنم."
        ]

        print("\nPerforming sentiment analysis on sample sentences...")
        results = predict_examples(
            loaded_model, loaded_tokenizer, sample_texts, device)

        # Display results
        print("\nSentiment Analysis Results:")
        print("-" * 50)
        for text, logits, pred in results:
            sentiment_labels = ["Negative", "Positive", "Neutral"]
            sentiment = sentiment_labels[pred] if pred < len(
                sentiment_labels) else f"Label {pred}"
            print(f"Text: {text}")
            print(f"Sentiment: {sentiment}")
            print(f"Scores: {logits}")
            print("-" * 30)

    else:
        print("Trained model not found. Starting training process...")

        # 1. Data preprocessing
        data_path = './data/sentipers.xlsx'
        (train_texts, train_labels), (val_texts, val_labels), (test_texts,
                                                               test_labels) = load_and_preprocess_data(data_path)

        # 2. Balance dataset using data augmentation
        print("\nBalancing dataset to handle class imbalance...")
        balanced_train_texts, balanced_train_labels, val_texts, val_labels, test_texts, test_labels = balance_dataset(
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
        )

        # 3. Create dataset and dataloader with balanced data
        train_dataset = SentimentDataset(
            balanced_train_texts, balanced_train_labels, tokenizer)
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
        test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        test_loader = DataLoader(test_dataset, batch_size=16)

        # 4. Train model with class weights for imbalanced data
        print("\nTraining with class weights to handle imbalanced data...")
        train_model(model, train_loader, val_loader, epochs=5,
                    lr=1e-5, train_labels=balanced_train_labels)

        # 5. Evaluate model
        predictions, true_labels = evaluate_model(model, test_loader, "Test")

        # 6. Visualize results
        visualize_results(predictions, true_labels)

        # 7. Save model
        os.makedirs('models', exist_ok=True)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Model saved to {model_path}.")

        # After training and evaluation
        sample_texts = [
            "این فیلم خیلی خوب بود و من لذت بردم.",
            "کیفیت محصول اصلا خوب نبود و ناراضی‌ام.",
            "قیمتش معمولی بود و تفاوتی با بقیه نداشت."
        ]
        predict_examples(model, tokenizer, sample_texts, device)


if __name__ == "__main__":
    main()
