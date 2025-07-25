import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm


class EnhancedPersianDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask=None):
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) with 1 for real tokens, 0 for pad
            # key_padding_mask: (batch, seq_len) with True for pad, False for real tokens
            key_padding_mask = (attention_mask == 0)
        attn_output, _ = self.attention(
            x, x, x, key_padding_mask=key_padding_mask)
        attn_output = self.dropout(attn_output)
        return self.norm(x + attn_output)


class EnhancedPersianSentimentModel(nn.Module):
    def __init__(self, model_name, num_classes=3, dropout_rate=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # Enhanced classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = AttentionLayer(self.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Apply attention mechanism
        attention_output = self.attention(sequence_output, attention_mask)

        # Global average pooling
        pooled_output = torch.mean(attention_output, dim=1)

        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class EnhancedTrainer:
    def __init__(self, model, tokenizer, device, class_weights=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.class_weights = class_weights

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(
                class_weights, dtype=torch.float32, device=device))
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader, optimizer, scheduler):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        progress_bar = tqdm(
            train_loader, desc='[Training] Batch', ncols=80, leave=True)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        return total_loss / len(train_loader), correct / total

    def evaluate(self, data_loader, phase='Eval'):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        progress_bar = tqdm(
            data_loader, desc=f'[{phase}] Batch', ncols=80, leave=True)
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        return total_loss / len(data_loader), all_predictions, all_labels


def train_enhanced_model(train_data, val_data, test_data, class_weights,
                         model_name='HooshvareLab/bert-fa-base-uncased',
                         output_dir='models/enhanced_model',
                         results_dir='results/improvements'):
    # Check if model already exists
    model_path = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(model_path):
        print(
            f"[Training] Model already exists at {model_path}. Skipping training.")
        # Load model and evaluate
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EnhancedPersianSentimentModel(
            model_name, num_classes=3, dropout_rate=0.3)
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        trainer = EnhancedTrainer(model, tokenizer, device, class_weights)
        # Prepare test set
        test_dataset = EnhancedPersianDataset(
            test_data[0], test_data[1], tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        print("[Training] Evaluating existing model on test set...")
        test_loss, test_preds, test_labels = trainer.evaluate(
            test_loader, phase='Test')
        class_names = ['Negative', 'Neutral', 'Positive']
        report = classification_report(
            test_labels, test_preds, target_names=class_names, output_dict=True)
        conf_matrix = confusion_matrix(test_labels, test_preds)
        # Save results
        os.makedirs(results_dir, exist_ok=True)
        with open(f'{results_dir}/enhanced_model_metrics.json', 'w') as f:
            json.dump(report, f, indent=2)
        create_confusion_matrix_plot(conf_matrix, class_names, results_dir)
        return model, report, conf_matrix

    print("[Training] Starting model training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = EnhancedPersianDataset(
        train_data[0], train_data[1], tokenizer)
    val_dataset = EnhancedPersianDataset(val_data[0], val_data[1], tokenizer)
    test_dataset = EnhancedPersianDataset(
        test_data[0], test_data[1], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model = EnhancedPersianSentimentModel(
        model_name, num_classes=3, dropout_rate=0.3)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 5
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)
    trainer = EnhancedTrainer(model, tokenizer, device, class_weights)
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(5):
        print(f"[Training] Epoch {epoch+1}/5")
        train_loss, train_acc = trainer.train_epoch(
            train_loader, optimizer, scheduler)
        val_loss, val_preds, val_labels = trainer.evaluate(
            val_loader, phase='Validation')
        val_acc = sum(1 for p, l in zip(val_preds, val_labels)
                      if p == l) / len(val_labels)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        print(
            f'[Training] Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(
                output_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[Training] Early stopping at epoch {epoch+1}')
                break
    print("[Training] Training complete. Evaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(
        output_dir, 'best_model.pth'), map_location=device))
    test_loss, test_preds, test_labels = trainer.evaluate(
        test_loader, phase='Test')
    class_names = ['Negative', 'Neutral', 'Positive']
    report = classification_report(
        test_labels, test_preds, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(test_labels, test_preds)
    os.makedirs(results_dir, exist_ok=True)
    with open(f'{results_dir}/enhanced_model_metrics.json', 'w') as f:
        json.dump(report, f, indent=2)
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }
    with open(f'{results_dir}/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    create_training_plots(history, results_dir)
    create_confusion_matrix_plot(conf_matrix, class_names, results_dir)
    print("[Training] All results, metrics, and plots are saved.")
    # Save misclassified samples for analysis
    print("[Analysis] Saving misclassified samples for further analysis...")
    # Load test texts
    test_texts = test_data[0]
    misclassified = []
    for text, true_label, pred_label in zip(test_texts, test_labels, test_preds):
        if true_label != pred_label:
            misclassified.append({
                'text': text,
                'true_label': class_names[true_label],
                'predicted_label': class_names[pred_label]
            })
    import pandas as pd
    pd.DataFrame(misclassified).to_csv(
        f'{results_dir}/misclassified_samples.csv', index=False)
    print(
        f"[Analysis] Saved {len(misclassified)} misclassified samples to {results_dir}/misclassified_samples.csv")
    return model, report, conf_matrix


def create_training_plots(history, results_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(history['train_accuracies'], label='Train Accuracy')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/training_plots.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def create_confusion_matrix_plot(conf_matrix, class_names, results_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Enhanced Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/confusion_matrix_enhanced.png',
                dpi=300, bbox_inches='tight')
    plt.close()
