import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from .setup_model import model, device


def evaluate_model(model, data_loader, phase="Test"):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, labels=[0, 1, 2])
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted')
    cm = confusion_matrix(true_labels, predictions)
    # Display results
    print(f"{phase} Results: Accuracy: {accuracy:.4f}")
    for i, label in enumerate(['Negative', 'Neutral', 'Positive']):
        print(
            f"{label}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}, Support: {support[i]}")
    print(
        f"Weighted: Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1: {weighted_f1:.4f}")
    # Save log
    with open('reports/eval_results.log', 'a', encoding='utf-8') as f:
        f.write(f"{phase} Results: Accuracy: {accuracy:.4f}\n")
        for i, label in enumerate(['Negative', 'Neutral', 'Positive']):
            f.write(
                f"{label}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}, Support: {support[i]}\n")
        f.write(
            f"Weighted: Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1: {weighted_f1:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n")
    return predictions, true_labels
