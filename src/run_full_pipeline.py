import os
from enhanced_preprocessing import EnhancedPersianPreprocessor
from enhanced_model import train_enhanced_model


def file_exists(path):
    return os.path.exists(path)


print('========== Persian Sentiment Analysis Pipeline =========')

# Step 1: Preprocessing and Data Preparation
print('\n[Pipeline] Step 1: Preprocessing and Data Preparation...')
preprocessed_train = 'data/processed_data/train_enhanced.csv'
preprocessed_val = 'data/processed_data/val_enhanced.csv'
preprocessed_test = 'data/processed_data/test_enhanced.csv'
preprocessed_stats = 'data/processed_data/dataset_stats.json'
if all(file_exists(p) for p in [preprocessed_train, preprocessed_val, preprocessed_test, preprocessed_stats]):
    print('[Pipeline] Preprocessed data found. Skipping preprocessing.')
    preprocessor = EnhancedPersianPreprocessor()
    dataset = preprocessor.prepare_enhanced_dataset(
        'data/sentipers.xlsx', output_dir='data/processed_data')
else:
    print('[Pipeline] No preprocessed data found. Running preprocessing...')
    preprocessor = EnhancedPersianPreprocessor()
    dataset = preprocessor.prepare_enhanced_dataset(
        'data/sentipers.xlsx', output_dir='data/processed_data')
print('[Pipeline] Preprocessing complete.')

train_data = dataset['train_data']
val_data = dataset['val_data']
test_data = dataset['test_data']
class_weights = dataset['class_weights']

# Step 2: Model Training and Evaluation
print('\n[Pipeline] Step 2: Model Training and Evaluation...')
model_path = 'models/enhanced_model/best_model.pth'
if file_exists(model_path):
    print('[Pipeline] Trained model found. Skipping training, running evaluation only.')
    model, report, conf_matrix = train_enhanced_model(
        train_data, val_data, test_data, class_weights,
        model_name='HooshvareLab/bert-fa-base-uncased',
        output_dir='models/enhanced_model',
        results_dir='results/improvements'
    )
else:
    print('[Pipeline] No trained model found. Starting training...')
    model, report, conf_matrix = train_enhanced_model(
        train_data, val_data, test_data, class_weights,
        model_name='HooshvareLab/bert-fa-base-uncased',
        output_dir='models/enhanced_model',
        results_dir='results/improvements'
    )
print('[Pipeline] Model training and evaluation complete.')

print('\n========== Pipeline Finished =========')
print('All results, metrics, and plots are saved in results/improvements and data/processed_data.')
