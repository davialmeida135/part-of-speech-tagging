import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

class POSTagger:
    def __init__(self, model=None, preprocessor=None, device=None):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.model:
            self.model.to(self.device)
    
    def create_data_loader(self, X, y, batch_size=32, shuffle=True):
        """Create PyTorch DataLoader"""
        # Calculate actual sequence lengths (excluding padding)
        lengths = []
        for seq in X:
            length = len(seq) - np.sum(seq == 0)  # Count non-padding tokens
            lengths.append(max(1, length))  # Ensure at least length 1
        
        # Convert to tensors
        X_tensor = torch.LongTensor(X)
        y_tensor = torch.LongTensor(y)
        lengths_tensor = torch.LongTensor(lengths)  # Keep as LongTensor, will convert to CPU in model
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor, lengths_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return dataloader
    
    def train(self, train_X, train_y, dev_X, dev_y, epochs=10, batch_size=32, 
          learning_rate=0.001, patience=3):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built yet!")
        
        # Create data loaders
        train_loader = self.create_data_loader(train_X, train_y, batch_size, shuffle=True)
        dev_loader = self.create_data_loader(dev_X, dev_y, batch_size, shuffle=False)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
        )
        
        # Training variables
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        print(f"Training on device: {self.device}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            
            for batch_X, batch_y, batch_lengths in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                # Don't move batch_lengths to device - let model handle CPU conversion
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X, batch_lengths)
                
                # Fix: Ensure both outputs and targets have exactly the same dimensions
                current_batch_size, current_seq_len = batch_y.size()
                
                # Make sure outputs match target dimensions exactly
                if outputs.size(1) != current_seq_len:
                    if outputs.size(1) > current_seq_len:
                        outputs = outputs[:, :current_seq_len, :]  # Trim outputs
                    else:
                        # Pad outputs if they're shorter (shouldn't happen but safe)
                        pad_length = current_seq_len - outputs.size(1)
                        padding = torch.zeros(outputs.size(0), pad_length, outputs.size(2), 
                                            device=outputs.device, dtype=outputs.dtype)
                        outputs = torch.cat([outputs, padding], dim=1)
                
                # Now reshape - dimensions should match
                outputs_flat = outputs.contiguous().view(-1, outputs.size(-1))
                targets_flat = batch_y.contiguous().view(-1)
                
                # Verify shapes match (remove this debug print after confirming fix)
                # print(f"Outputs shape: {outputs_flat.shape}, Targets shape: {targets_flat.shape}")
                
                # Calculate loss
                loss = criterion(outputs_flat, targets_flat)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
            
            # Validation phase
            val_loss, val_accuracy = self.evaluate(dev_X, dev_y, batch_size)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            avg_train_loss = train_loss / train_steps
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val Accuracy: {val_accuracy:.4f}')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping after {epoch+1} epochs')
                    # Load best model
                    self.model.load_state_dict(torch.load('best_model.pth'))
                    break
        
        return history

    def evaluate(self, X, y, batch_size=32):
        """Evaluate the model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        test_loader = self.create_data_loader(X, y, batch_size, shuffle=False)
        
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch_X, batch_y, batch_lengths in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                # Don't move batch_lengths to device
                
                # Forward pass
                outputs = self.model(batch_X, batch_lengths)
                
                # Ensure outputs and targets have the same sequence length
                batch_size, seq_len = batch_y.size()
                outputs = outputs[:, :seq_len, :]  # Trim outputs to match target length
                
                # Calculate loss
                outputs_flat = outputs.contiguous().view(-1, outputs.size(-1))
                targets_flat = batch_y.contiguous().view(-1)
                loss = criterion(outputs_flat, targets_flat)
                total_loss += loss.item()
                
                # Calculate accuracy (excluding padding tokens)
                predictions = torch.argmax(outputs, dim=-1)
                mask = (batch_y != 0)  # Non-padding tokens
                correct = (predictions == batch_y) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        
        return avg_loss, accuracy

    def predict(self, X, batch_size=32):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        predictions = []
        
        # Create data loader for predictions
        lengths = []
        for seq in X:
            length = len(seq) - np.sum(seq == 0)
            lengths.append(max(1, length))
        
        X_tensor = torch.LongTensor(X)
        lengths_tensor = torch.LongTensor(lengths)
        dataset = TensorDataset(X_tensor, lengths_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_X, batch_lengths in dataloader:
                batch_X = batch_X.to(self.device)
                # Don't move batch_lengths to device
                
                outputs = self.model(batch_X, batch_lengths)
                
                # Ensure outputs match input length
                seq_len = batch_X.size(1)
                outputs = outputs[:, :seq_len, :]  # Trim to input length
                
                batch_predictions = torch.argmax(outputs, dim=-1)
                predictions.append(batch_predictions.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def save_model(self, path):
        """Save the model to a file"""
        if self.model is None:
            raise ValueError("Model not built yet!")
        
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")