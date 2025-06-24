import tensorflow as tf
from keras import layers
import numpy as np
import pickle
import keras
from preprocessing import POSDataPreprocessor
import pandas as pd
from rnn_model import RNNModel
import os
from engine import POSTagger

def configure_gpu(try_gpu=True):
    """Configure GPU settings for TensorFlow"""
    print("Configuring GPU...")
    
    if not try_gpu:
        """Configure GPU settings for TensorFlow"""
        print("Configuring for CPU training...")
        
        # Force CPU usage by setting CUDA_VISIBLE_DEVICES to empty
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Disable GPU devices
        tf.config.set_visible_devices([], 'GPU')
        
        print("GPU disabled - using CPU for training")
        print(f"TensorFlow version: {tf.__version__}")
        
        return False
    
    # Check if GPU is available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision for faster training
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled")
            
            print(f"GPU(s) found and configured: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found. Training will use CPU.")
        print("To enable GPU support:")
        print("  1. Install CUDA toolkit")
        print("  2. pip install tensorflow[and-cuda]==2.19.0")
    
    # Print TensorFlow build info for debugging
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    return len(gpus) > 0

def create_result_dataset(predictions, true_labels, sequences, preprocessor, output_path):
    """Create dataset in the same format as your probabilistic models"""
    results = []
    
    for i, (pred_seq, true_seq, word_seq) in enumerate(zip(predictions, true_labels, sequences)):
        for j, (pred_tag_idx, true_tag_idx, word_idx) in enumerate(zip(pred_seq, true_seq, word_seq)):
            # Skip padding tokens
            if word_idx == 0:  # <PAD>
                continue
            
            word = preprocessor.idx2word.get(word_idx, '<UNK>')
            true_tag = preprocessor.idx2tag.get(true_tag_idx, '<PAD>')
            pred_tag = preprocessor.idx2tag.get(pred_tag_idx, '<PAD>')
            
            # Only add non-padding entries
            if true_tag != '<PAD>' and pred_tag != '<PAD>':
                results.append({
                    'id': i,
                    'word': word,
                    'real': true_tag,
                    'pred': pred_tag
                })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path} with {len(df)} examples")
    return df

def main():
    """Main training and evaluation script"""
    #gpu_available = configure_gpu(True)
    
    # Set paths relative to project root
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load preprocessor and data
    print("Loading data...")
    preprocessor = POSDataPreprocessor()
    vocab_path = os.path.join(base_path, 'data', 'models', 'vocab.pkl')
    data_path = os.path.join(base_path, 'data', 'models', 'pos_data.npz')
    
    preprocessor.load_vocab(vocab_path)
    data = np.load(data_path)
    
    train_X, train_y = data['train_X'], data['train_y']
    dev_X, dev_y = data['dev_X'], data['dev_y']
    test_X, test_y = data['test_X'], data['test_y']
    
    print(f"Train shape: {train_X.shape}, {train_y.shape}")
    print(f"Dev shape: {dev_X.shape}, {dev_y.shape}")
    print(f"Test shape: {test_X.shape}, {test_y.shape}")
    
    # Initialize model
    vocab_size = len(preprocessor.word2idx)
    tag_size = len(preprocessor.tag2idx)
    
    print(f"Vocab size: {vocab_size}, Tag size: {tag_size}")

    model = RNNModel(
        vocab_size=vocab_size,
        tag_size=tag_size,
        embedding_dim=100,
        hidden_dim=128
    )
    
    tagger = POSTagger(
        model=model.build_model(),
        preprocessor=preprocessor
    )
    
    print("Model summary:")
    tagger.model.summary()
    
    print("Training model...")
    history = tagger.train(
        train_X, train_y,
        dev_X, dev_y,
        epochs=15,
        batch_size=256
    )
    
    # Evaluate on test set (using the actual test set, not dev)
    print("Evaluating on test set...")
    test_loss, test_accuracy = tagger.evaluate(test_X, test_y)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Make predictions and create run dataset
    print("Creating predictions...")
    test_predictions = tagger.predict(test_X)
    
    # Create run dataset compatible with your analysis notebooks
    output_path = os.path.join(base_path, 'data', 'runs', 'rnn_test.csv')
    run_df = create_result_dataset(
        test_predictions, test_y, test_X, 
        preprocessor, output_path
    )
    
    print("RNN training and evaluation complete!")
    
    # Save model
    model_path = os.path.join(base_path, 'data', 'models', 'rnn_pos_model')
    tagger.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()