import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.3):
        """
        Initialize the RNN model with parameters.
        
        :param vocab_size: Size of the vocabulary
        :param tag_size: Number of unique tags for classification
        :param embedding_dim: Dimension of the embedding layer
        :param hidden_dim: Dimension of the hidden layer in LSTM
        :param num_layers: Number of LSTM layers
        :param dropout: Dropout rate
        """
        super(RNNModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # <PAD> token index
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer (bidirectional LSTM outputs hidden_dim * 2)
        self.classifier = nn.Linear(hidden_dim * 2, tag_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize embedding weights
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        # Set padding token embedding to zero
        nn.init.constant_(self.embedding.weight[0], 0)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        
    def forward(self, x, lengths=None):
        """
        Forward pass
        
        :param x: Input tensor of shape (batch_size, seq_len)
        :param lengths: Actual lengths of sequences (optional)
        :return: Output tensor of shape (batch_size, seq_len, tag_size)
        """
        batch_size, seq_len = x.size()
        
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # For simplicity, don't use packed sequences to avoid length mismatches
        # The masking will be handled by the loss function (ignore_index=0)
        
        # LSTM layer
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Apply dropout
        lstm_out = self.dropout_layer(lstm_out)
        
        # Classification layer
        output = self.classifier(lstm_out)  # (batch_size, seq_len, tag_size)
        
        return output
    
    def predict(self, x, lengths=None):
        """
        Make predictions (returns class indices)
        
        :param x: Input tensor
        :param lengths: Actual lengths of sequences (optional)
        :return: Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, lengths)
            predictions = torch.argmax(logits, dim=-1)
        return predictions