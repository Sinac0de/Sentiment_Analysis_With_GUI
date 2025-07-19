import torch
from torch.utils.data import Dataset, DataLoader
from .preprocess_data import load_and_preprocess_data
from .setup_model import tokenizer


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def get_dataloaders(data_path='../data/sentipers.xlsx', batch_size=16, subset_size=None):
    print("Loading and preprocessing data...")
    # Load data
    (train_texts, train_labels), (val_texts, val_labels), (test_texts,
                                                           test_labels) = load_and_preprocess_data(data_path)

    # If subset_size is specified, use only a portion of the data
    if subset_size:
        print(f"Using subset of {subset_size} samples for faster training...")
        train_texts = train_texts[:subset_size]
        train_labels = train_labels[:subset_size]
        val_texts = val_texts[:subset_size//2]
        val_labels = val_labels[:subset_size//2]

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")

    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        subset_size=1000)  # Only 1000 samples for testing
    print("Dataloaders created successfully!")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
