import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('HooshvareLab/bert-fa-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    'HooshvareLab/bert-fa-base-uncased', num_labels=3)
model.to(device)
