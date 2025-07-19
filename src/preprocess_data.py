import pandas as pd
from PersianStemmer import PersianStemmer
from sklearn.model_selection import train_test_split
import os
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_excel(file_path)

    # Check columns
    print("File columns:", df.columns)

    # Check for null values in text column
    if df['text'].isnull().any():
        print(
            "Warning: Some values in 'text' column are null. Replacing with empty string...")
        df['text'] = df['text'].fillna('')

    # Convert labels
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

    # Text normalization
    stemmer = PersianStemmer()
    df['text'] = df['text'].apply(lambda x: stemmer.run(
        str(x).replace('ك', 'ک').replace('ي', 'ی')) if x else x)

    # Remove stopwords and punctuation
    # Persian stopwords list (sample, can be expanded)
    persian_stopwords = set(['و', 'در', 'به', 'از', 'که', 'این', 'را', 'با', 'برای', 'است', 'آن', 'یک', 'تا',
                            'می', 'بر', 'اما', 'یا', 'هم', 'نیز', 'شود', 'کرد', 'شد', 'های', 'خود', 'بود', 'کرده', 'ای', 'ها'])

    def clean_text(text):
        # Remove punctuation
        text = re.sub(r'[\W_]+', ' ', text)
        # Remove stopwords
        text = ' '.join(
            [w for w in text.split() if w not in persian_stopwords])
        return text
    df['text'] = df['text'].apply(lambda x: clean_text(x) if x else x)

    # Remove unnecessary columns
    df = df[['text', 'label']]

    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['text'], df['label'], test_size=0.3, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )

    # Save preprocessed data
    os.makedirs('../data/processed_data', exist_ok=True)
    pd.DataFrame({'text': train_texts, 'label': train_labels}).to_csv(
        '../data/processed_data/train.csv', index=False)
    pd.DataFrame({'text': val_texts, 'label': val_labels}).to_csv(
        '../data/processed_data/val.csv', index=False)
    pd.DataFrame({'text': test_texts, 'label': test_labels}).to_csv(
        '../data/processed_data/test.csv', index=False)

    # Check and save label distribution and sample data
    label_counts = df['label'].value_counts().sort_index()
    os.makedirs('../reports', exist_ok=True)
    with open('../reports/label_distribution.log', 'w', encoding='utf-8') as f:
        f.write('Label distribution (0=Negative, 1=Neutral, 2=Positive):\n')
        for label, count in label_counts.items():
            f.write(f'Label {label}: {count}\n')
        f.write('\nSample data:\n')
        for i, row in df.head(10).iterrows():
            f.write(f"Text: {row['text']} | Label: {row['label']}\n")

    return (train_texts.tolist(), train_labels.tolist()), (val_texts.tolist(), val_labels.tolist()), (test_texts.tolist(), test_labels.tolist())


if __name__ == "__main__":
    file_path = '../data/sentipers.xlsx'
    (train_texts, train_labels), (val_texts, val_labels), (test_texts,
                                                           test_labels) = load_and_preprocess_data(file_path)
