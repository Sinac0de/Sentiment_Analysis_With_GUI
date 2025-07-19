import pandas as pd
import numpy as np
from sklearn.utils import resample
import random
import re


def augment_negative_samples(texts, labels, augmentation_factor=2):
    """
    Augment negative samples to balance the dataset

    Args:
        texts: List of text samples
        labels: List of corresponding labels
        augmentation_factor: How many times to increase negative samples

    Returns:
        augmented_texts, augmented_labels
    """

    # Separate negative samples
    negative_texts = [text for text, label in zip(texts, labels) if label == 0]
    negative_labels = [0] * len(negative_texts)

    print(f"Original negative samples: {len(negative_texts)}")

    if len(negative_texts) == 0:
        print("No negative samples found!")
        return texts, labels

    # Calculate how many samples we need to add
    target_count = len(negative_texts) * augmentation_factor
    samples_to_add = target_count - len(negative_texts)

    if samples_to_add <= 0:
        print("No augmentation needed")
        return texts, labels

    # Simple augmentation techniques for Persian text
    augmented_negative_texts = []
    augmented_negative_labels = []

    for _ in range(samples_to_add):
        # Randomly select a negative sample
        original_text = random.choice(negative_texts)

        # Apply augmentation techniques
        augmented_text = apply_augmentation(original_text)

        augmented_negative_texts.append(augmented_text)
        augmented_negative_labels.append(0)

    # Combine original and augmented data
    all_texts = texts + augmented_negative_texts
    all_labels = labels + augmented_negative_labels

    print(f"Added {len(augmented_negative_texts)} augmented negative samples")
    print(f"Total samples after augmentation: {len(all_texts)}")

    return all_texts, all_labels


def apply_augmentation(text):
    """
    Apply simple augmentation techniques to Persian text
    """
    augmented_text = text

    # Technique 1: Add common negative words
    negative_words = ['بد', 'ضعیف', 'نامناسب', 'ناراضی', 'مشکل', 'خطا']
    if random.random() < 0.3:  # 30% chance
        word = random.choice(negative_words)
        augmented_text = f"{word} {augmented_text}"

    # Technique 2: Add negative prefixes
    negative_prefixes = ['اصلاً', 'هیچ', 'کلاً', 'اصلاً']
    if random.random() < 0.2:  # 20% chance
        prefix = random.choice(negative_prefixes)
        augmented_text = f"{prefix} {augmented_text}"

    # Technique 3: Add negative suffixes
    negative_suffixes = ['نیست', 'نبود', 'نداشت']
    if random.random() < 0.15:  # 15% chance
        suffix = random.choice(negative_suffixes)
        augmented_text = f"{augmented_text} {suffix}"

    # Technique 4: Synonym replacement (simple)
    synonyms = {
        'خوب': 'مناسب',
        'عالی': 'عالی',
        'مشکل': 'مسئله',
        'بد': 'ضعیف',
        'ناراضی': 'ناراحت'
    }

    for original, synonym in synonyms.items():
        if original in augmented_text and random.random() < 0.1:  # 10% chance
            augmented_text = augmented_text.replace(original, synonym)

    return augmented_text


def balance_dataset(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels):
    """
    Balance the entire dataset by augmenting minority classes
    """
    print("=== Dataset Balancing ===")

    # Count original distribution
    train_counts = pd.Series(train_labels).value_counts().sort_index()
    print("Original training distribution:")
    sentiment_names = ['Negative', 'Neutral', 'Positive']
    for i, count in train_counts.items():
        print(f"  {sentiment_names[i]}: {count}")

    # Find the majority class count
    max_count = train_counts.max()

    # Balance each class to match the majority
    balanced_train_texts = []
    balanced_train_labels = []

    for label in [0, 1, 2]:  # Negative, Neutral, Positive
        class_texts = [text for text, lab in zip(
            train_texts, train_labels) if lab == label]
        class_labels = [label] * len(class_texts)

        if len(class_texts) < max_count:
            # Need to augment this class
            if label == 0:  # Negative class
                # Use augmentation for negative samples
                augmented_texts, augmented_labels = augment_negative_samples(
                    class_texts, class_labels,
                    augmentation_factor=max_count // len(class_texts) + 1
                )
                # Take only the required number
                balanced_train_texts.extend(augmented_texts[:max_count])
                balanced_train_labels.extend(augmented_labels[:max_count])
            else:
                # Use oversampling for other classes
                if len(class_texts) > 0:
                    oversampled_texts = resample(
                        class_texts,
                        n_samples=max_count,
                        random_state=42
                    )
                    balanced_train_texts.extend(oversampled_texts)
                    balanced_train_labels.extend([label] * max_count)
        else:
            # This class has enough samples, just take max_count
            balanced_train_texts.extend(class_texts[:max_count])
            balanced_train_labels.extend(class_labels[:max_count])

    # Shuffle the balanced dataset
    combined = list(zip(balanced_train_texts, balanced_train_labels))
    random.shuffle(combined)
    balanced_train_texts, balanced_train_labels = zip(*combined)

    print("\nBalanced training distribution:")
    balanced_counts = pd.Series(
        balanced_train_labels).value_counts().sort_index()
    for i, count in balanced_counts.items():
        print(f"  {sentiment_names[i]}: {count}")

    return list(balanced_train_texts), list(balanced_train_labels), val_texts, val_labels, test_texts, test_labels


if __name__ == "__main__":
    # Test augmentation
    test_texts = ["این محصول خیلی بد است", "کیفیت مناسب نیست"]
    test_labels = [0, 0]

    augmented_texts, augmented_labels = augment_negative_samples(
        test_texts, test_labels, 2)

    print("Original texts:", test_texts)
    print("Augmented texts:", augmented_texts)
