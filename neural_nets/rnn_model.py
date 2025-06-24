import keras
from keras import layers

class RNNModel:
    def __init__(self, vocab_size, tag_size, embedding_dim=100, hidden_dim=128):
        """
        Initialize the RNN model with parameters.
        
        :param vocab_size: Size of the vocabulary
        :param tag_size: Number of unique tags for classification
        :param embedding_dim: Dimension of the embedding layer
        :param hidden_dim: Dimension of the hidden layer in LSTM
        """
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.model = None
        self.preprocessor = None

    def build_model(self):
        """Build the RNN model"""
        model = keras.Sequential([
                    # Embedding layer
                    layers.Embedding(
                        input_dim=self.vocab_size,
                        output_dim=self.embedding_dim,
                        mask_zero=False, 
                        name='embedding'
                    ),
                    
                    # Bidirectional LSTM
                    layers.Bidirectional(
                        layers.LSTM(
                            self.hidden_dim,
                            return_sequences=True,
                            dropout=0.3,
                            recurrent_dropout=0.3
                        ),
                        name='bi_lstm'
                    ),
                    
                    # Dense layer for classification
                    layers.TimeDistributed(
                        layers.Dense(self.tag_size, activation='softmax'),
                        name='output'
                    )
                ])
                
                # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model