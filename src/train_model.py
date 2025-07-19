import torch
from torch.optim import AdamW
from .setup_model import model, device
from .dataset import get_dataloaders
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def calculate_class_weights(train_labels):
    """Calculate class weights to handle imbalanced data"""
    # Convert to numpy array if it's a list
    if isinstance(train_labels, list):
        train_labels = np.array(train_labels)

    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )

    # Convert to tensor
    class_weights = torch.FloatTensor(class_weights).to(device)

    print("Class weights calculated:")
    sentiment_names = ['Negative', 'Neutral', 'Positive']
    for i, weight in enumerate(class_weights):
        print(f"  {sentiment_names[i]}: {weight:.3f}")

    return class_weights


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-5, patience=2, train_labels=None):
    import copy
    from sklearn.metrics import accuracy_score

    # Calculate class weights if train_labels are provided
    if train_labels is not None:
        class_weights = calculate_class_weights(train_labels)
        # Use weighted loss
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()
    best_val_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        print(f"\nEpoch {epoch+1}/{epochs}")
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

        # Evaluate on validation data
        from .evaluate_model import evaluate_model
        predictions, true_labels = evaluate_model(
            model, val_loader, "Validation")
        val_acc = accuracy_score(true_labels, predictions)
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            model.save_pretrained('models/best_model_earlystop')
            patience_counter = 0
            print("Best model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best weights
    model.load_state_dict(best_model_wts)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    print("Loading data...")
    # Use subset for faster training
    train_loader, val_loader, _ = get_dataloaders(
        subset_size=2000)  # Only 2000 samples
    print(f"Training on {len(train_loader)} batches")
    print(f"Validation on {len(val_loader)} batches")

    train_model(model, train_loader, val_loader)
