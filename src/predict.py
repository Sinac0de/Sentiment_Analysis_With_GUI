import torch
from .setup_model import model, tokenizer, device


def predict_sentiment(text):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt',
                         max_length=128, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return ['Negative', 'Neutral', 'Positive'][pred]


if __name__ == "__main__":
    sample_text = "عاشقشم. شاهکاره"
    print(f"Text: {sample_text}")
    print(f"Sentiment: {predict_sentiment(sample_text)}")
