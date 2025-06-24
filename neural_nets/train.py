import tensorflow as tf
from keras import layers
import numpy as np
import pickle
import keras
from preprocessing import POSDataPreprocessor
import pandas as pd
from rnn_model import RNNModel
import os

def configure_gpu():
    """Configure GPU settings for TensorFlow"""
    print("Configuring GPU...")
    
    # Check if GPU is available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
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

class POSTagger:
    def __init__(self, model=None, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
    
    def train(self, train_X, train_y, dev_X, dev_y, epochs=10, batch_size=32):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built yet!")
        
        # Create callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_X, train_y,
            validation_data=(dev_X, dev_y),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=-1)
    
    def evaluate(self, X, y):
        """Evaluate model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        return self.model.evaluate(X, y, verbose=0)
    
    def save_model(self, path):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        self.model.save(path)
    
    def load_model(self, path):
        """Load a trained model"""
        self.model = keras.models.load_model(path)

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
    gpu_available = configure_gpu()
    
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
        batch_size=32 if gpu_available else 16  # Smaller batch size for CPU
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