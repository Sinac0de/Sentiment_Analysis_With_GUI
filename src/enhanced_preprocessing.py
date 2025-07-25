import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import re
import random
import json
import os
from hazm import Normalizer, Stemmer, word_tokenize
from tqdm import tqdm


class EnhancedPersianPreprocessor:
    def __init__(self):
        self.hazm_normalizer = Normalizer()
        self.hazm_stemmer = Stemmer()
        self.stop_words = set()
        self.load_persian_stopwords()

    def load_persian_stopwords(self):
        persian_stopwords = [
            'و', 'در', 'به', 'از', 'که', 'این', 'است', 'را', 'با', 'برای',
            'آن', 'یک', 'خود', 'تا', 'بر', 'بود', 'شد', 'شدند', 'خواهد',
            'می', 'های', 'ها', 'هم', 'یا', 'اما', 'اگر', 'چون', 'چرا'
        ]
        self.stop_words = set(persian_stopwords)

    def normalize_persian_text(self, text: str) -> str:
        if pd.isna(text) or text == '':
            return ''

        text = str(text)

        # hazm normalization
        text = self.hazm_normalizer.normalize(text)

        # Character normalization (extra)
        char_mapping = {
            'ك': 'ک', 'ي': 'ی', 'ة': 'ه', 'ؤ': 'و', 'إ': 'ا', 'أ': 'ا'
        }

        for old_char, new_char in char_mapping.items():
            text = text.replace(old_char, new_char)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def random_deletion(self, words, p=0.15):
        if len(words) == 1:
            return words
        new_words = []
        for word in words:
            if random.uniform(0, 1) > p:
                new_words.append(word)
        if len(new_words) == 0:
            return [random.choice(words)]
        return new_words

    def random_swap(self, words, n=1):
        words = words.copy()
        for _ in range(n):
            idx1 = random.randint(0, len(words)-1)
            idx2 = random.randint(0, len(words)-1)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return words

    def replace_negative_synonyms(self, text):
        negative_synonyms = {
            'بد': ['نامناسب', 'ضعیف', 'نادرست'],
            'ضعیف': ['بد', 'نامناسب', 'کم'],
            'ناراضی': ['ناراضی', 'ناراحت', 'مخالف'],
            'مشکل': ['مشکل', 'دشواری', 'سختی']
        }

        words = word_tokenize(text)
        for i, word in enumerate(words):
            if word in negative_synonyms:
                words[i] = random.choice(negative_synonyms[word])

        return ' '.join(words)

    def augment_negative_samples(self, texts, labels, augmentation_factor=3):
        negative_texts = [text for text, label in zip(
            texts, labels) if label == 0]
        negative_labels = [0] * len(negative_texts)

        augmented_texts = []
        augmented_labels = []

        for text in negative_texts:
            # Original
            augmented_texts.append(text)
            augmented_labels.append(0)
            words = word_tokenize(text)
            # Synonym replacement
            syn_text = self.replace_negative_synonyms(text)
            if syn_text != text:
                augmented_texts.append(syn_text)
                augmented_labels.append(0)
            # Random deletion
            del_text = ' '.join(self.random_deletion(words, p=0.2))
            if del_text != text:
                augmented_texts.append(del_text)
                augmented_labels.append(0)
            # Random swap
            swap_text = ' '.join(self.random_swap(words, n=1))
            if swap_text != text:
                augmented_texts.append(swap_text)
                augmented_labels.append(0)
        return augmented_texts, augmented_labels

    def preprocess_text(self, text, remove_stopwords=True):
        text = self.normalize_persian_text(text)

        words = word_tokenize(text)

        if remove_stopwords:
            words = [word for word in words if word not in self.stop_words]

        # hazm stemming only
        stemmed_words = [self.hazm_stemmer.stem(word) for word in words]
        text = ' '.join(stemmed_words)

        return text.strip()

    def prepare_enhanced_dataset(self, data_path, output_dir='data/processed_data'):
        # Check if processed files exist
        train_path = os.path.join(output_dir, 'train_enhanced.csv')
        val_path = os.path.join(output_dir, 'val_enhanced.csv')
        test_path = os.path.join(output_dir, 'test_enhanced.csv')
        stats_path = os.path.join(output_dir, 'dataset_stats.json')
        if all(os.path.exists(p) for p in [train_path, val_path, test_path, stats_path]):
            print(
                f"[Preprocessing] Processed files found in {output_dir}. Skipping preprocessing.")
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            return {
                'train_data': (train_df['text'].tolist(), train_df['label'].tolist()),
                'val_data': (val_df['text'].tolist(), val_df['label'].tolist()),
                'test_data': (test_df['text'].tolist(), test_df['label'].tolist()),
                'class_weights': np.array(stats['class_weights']),
                'stats': stats
            }
        print(
            f"[Preprocessing] No processed files found. Starting preprocessing from {data_path} ...")
        df = pd.read_excel(data_path)
        print(f"[Preprocessing] Loaded {len(df)} samples from Excel.")

        def convert_label(label):
            if label in [-2, -1]:
                return 0
            elif label == 0:
                return 1
            else:
                return 2
        print("[Preprocessing] Converting polarity to label ...")
        df['label'] = df['polarity'].apply(convert_label)
        print("[Preprocessing] Normalizing and stemming texts ...")
        df['processed_text'] = list(tqdm(df['text'], desc='Normalizing', ncols=80, mininterval=0.5,
                                    maxinterval=2.0, smoothing=0.1, leave=True, unit='sample', colour='green', total=len(df)))
        df['processed_text'] = df['processed_text'].apply(self.preprocess_text)
        print("[Preprocessing] Augmenting negative samples ...")
        negative_texts, negative_labels = self.augment_negative_samples(
            df['processed_text'].tolist(), df['label'].tolist()
        )
        augmented_df = pd.DataFrame({
            'text': negative_texts,
            'label': negative_labels
        })
        original_df = df[['processed_text', 'label']].rename(
            columns={'processed_text': 'text'})
        combined_df = pd.concat([original_df, augmented_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['text'])
        X = combined_df['text'].tolist()
        y = combined_df['label'].tolist()
        print("[Preprocessing] Splitting data ...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        os.makedirs(output_dir, exist_ok=True)
        train_df = pd.DataFrame({'text': X_train, 'label': y_train})
        val_df = pd.DataFrame({'text': X_val, 'label': y_val})
        test_df = pd.DataFrame({'text': X_test, 'label': y_test})
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        stats = {
            'original_samples': len(df),
            'augmented_samples': len(combined_df),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'class_distribution': {
                'negative': sum(1 for label in y if label == 0),
                'neutral': sum(1 for label in y if label == 1),
                'positive': sum(1 for label in y if label == 2)
            },
            'class_weights': class_weights.tolist()
        }
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"[Preprocessing] Done. Processed data saved in {output_dir}.")
        return {
            'train_data': (X_train, y_train),
            'val_data': (X_val, y_val),
            'test_data': (X_test, y_test),
            'class_weights': class_weights,
            'stats': stats
        }


if __name__ == "__main__":
    preprocessor = EnhancedPersianPreprocessor()
    dataset = preprocessor.prepare_enhanced_dataset('data/sentipers.xlsx')
