import keras
import numpy as np

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