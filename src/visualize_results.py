import matplotlib.pyplot as plt
import seaborn as sns
from .evaluate_model import confusion_matrix


def visualize_results(predictions, true_labels):
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
                'Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Sentiment distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=predictions, palette='viridis')
    plt.xticks(ticks=[0, 1, 2], labels=['Negative', 'Neutral', 'Positive'])
    plt.title('Predicted Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()
