import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def analyze_data_distribution():
    # Load the original dataset
    df = pd.read_excel('./data/sentipers.xlsx')

    print("=== Original Dataset Analysis ===")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Check polarity distribution
    polarity_counts = df['polarity'].value_counts().sort_index()
    print("\nPolarity distribution:")
    for polarity, count in polarity_counts.items():
        percentage = (count / len(df)) * 100
        print(f"Polarity {polarity}: {count} samples ({percentage:.1f}%)")

    # Map polarities to labels
    def map_label(polarity):
        if polarity in [-2, -1]:
            return 0  # Negative
        elif polarity == 0:
            return 1  # Neutral
        elif polarity in [1, 2]:
            return 2  # Positive
        else:
            return 1  # Default: Neutral

    df['label'] = df['polarity'].apply(map_label)

    # Check label distribution
    label_counts = df['label'].value_counts().sort_index()
    print("\nLabel distribution:")
    sentiment_names = ['Negative', 'Neutral', 'Positive']
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        sentiment = sentiment_names[label]
        print(f"{sentiment} (Label {label}): {count} samples ({percentage:.1f}%)")

    # Visualize distribution
    plt.figure(figsize=(12, 5))

    # Original polarity distribution
    plt.subplot(1, 2, 1)
    polarity_counts.plot(kind='bar', color='skyblue')
    plt.title('Original Polarity Distribution')
    plt.xlabel('Polarity')
    plt.ylabel('Count')
    plt.xticks(rotation=0)

    # Label distribution
    plt.subplot(1, 2, 2)
    label_counts.plot(kind='bar', color='lightcoral')
    plt.title('Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(range(len(sentiment_names)), sentiment_names, rotation=0)

    plt.tight_layout()
    plt.savefig('reports/data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate imbalance ratio
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count
    print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}")

    if imbalance_ratio > 2:
        print("⚠️  Dataset is imbalanced! Consider using techniques like:")
        print("   - Class weights")
        print("   - Data augmentation")
        print("   - Oversampling/Undersampling")
        print("   - Focal Loss")

    return df, label_counts


if __name__ == "__main__":
    df, label_counts = analyze_data_distribution()
