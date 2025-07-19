import torch
from torch.optim import AdamW
from .setup_model import model, device
from .dataset import get_dataloaders
from tqdm import tqdm


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-5, patience=2):
    import copy
    from sklearn.metrics import accuracy_score
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
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
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
