import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import pickle

class POSDataPreprocessor:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.tag2idx = {'<PAD>': 0}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.idx2tag = {0: '<PAD>'}
        self.max_seq_length = 128
        
    def build_vocab(self, data_path):
        """Build vocabulary from training data"""
        word_counts = Counter()
        tag_counts = Counter()
        
        # Conta as frequências de palavras e tags
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                tokens = line.split()
                for token in tokens:
                    if '_' not in token:
                        continue
                    word, tag = token.rsplit('_', 1)
                    
                    # Handle numeric words like in your extractors
                    try:
                        float(word)
                        word = 'numeric-word'
                    except Exception:
                        pass
                        
                    word_counts[word] += 1
                    tag_counts[tag] += 1
        
        # Build word vocabulary (keep words with freq > 1, others become <UNK>)
        word_idx = 2  # Start after <PAD> and <UNK>
        for word, count in word_counts.items():
            if count > 1:  # Similar to your unk-word handling
                self.word2idx[word] = word_idx
                self.idx2word[word_idx] = word
                word_idx += 1
        
        # Build tag vocabulary
        tag_idx = 1  # Start after <PAD>
        for tag in tag_counts:
            self.tag2idx[tag] = tag_idx
            self.idx2tag[tag_idx] = tag
            tag_idx += 1
            
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Tag set size: {len(self.tag2idx)}")

        # Create vocabulary CSV for inspection
        vocab_df = pd.DataFrame([
            {'type': 'word', 'token': word, 'index': idx, 'frequency': word_counts.get(word, 0)}
            for word, idx in self.word2idx.items()
        ] + [
            {'type': 'tag', 'token': tag, 'index': idx, 'frequency': tag_counts.get(tag, 0)}
            for tag, idx in self.tag2idx.items()
        ])

        vocab_df.to_csv('neural_nets/vocabulary/vocabulary.csv', index=False)
        print("Saved vocabulary to neural_nets/vocabulary/vocabulary.csv")
        
    def text_to_sequences(self, data_path):
        """Convert text to sequences of indices"""
        sequences = []
        tag_sequences = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                tokens = line.split()
                word_seq = []
                tag_seq = []
                
                for token in tokens:
                    if '_' not in token:
                        continue
                    word, tag = token.rsplit('_', 1)
                    
                    # Handle numeric words
                    try:
                        float(word)
                        word = 'numeric-word'
                    except Exception:
                        pass
                    
                    # Convert to indices
                    word_idx = self.word2idx.get(word, self.word2idx['<UNK>'])
                    tag_idx = self.tag2idx.get(tag, 0)  # Should not happen with training data
                    
                    word_seq.append(word_idx)
                    tag_seq.append(tag_idx)
                
                if word_seq:  # Only add non-empty sequences
                    sequences.append(word_seq)
                    tag_sequences.append(tag_seq)
        
        return sequences, tag_sequences
    
    def pad_sequences(self, sequences, max_length=None):
        """Pad sequences to same length"""
        if max_length is None:
            max_length = self.max_seq_length
            
        padded = []
        for seq in sequences:
            if len(seq) > max_length:
                padded.append(seq[:max_length])
            else:
                padded.append(seq + [0] * (max_length - len(seq)))
        
        return np.array(padded)
    
    def save_vocab(self, path):
        """Save vocabulary mappings"""
        vocab_data = {
            'word2idx': self.word2idx,
            'tag2idx': self.tag2idx,
            'idx2word': self.idx2word,
            'idx2tag': self.idx2tag,
            'max_seq_length': self.max_seq_length
        }
        with open(path, 'wb') as f:
            pickle.dump(vocab_data, f)
    
    def load_vocab(self, path):
        """Load vocabulary mappings"""
        with open(path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.tag2idx = vocab_data['tag2idx']
        self.idx2word = vocab_data['idx2word']
        self.idx2tag = vocab_data['idx2tag']
        self.max_seq_length = vocab_data['max_seq_length']

    def save_readable_data(self, sequences, tag_sequences, output_path):
        """Save sequences in a readable CSV format"""
        readable_data = []
        
        for i, (word_seq, tag_seq) in enumerate(zip(sequences, tag_sequences)):
            # Convert indices back to words/tags
            words = [self.idx2word.get(idx, '<UNK>') for idx in word_seq if idx != 0] 
            tags = [self.idx2tag.get(idx, '<PAD>') for idx in tag_seq if idx != 0] 
            
            # Create sentence string
            sentence = ' '.join([f"{word}_{tag}" for word, tag in zip(words, tags)])
            
            readable_data.append({
                'sentence_id': i,
                'length': len(words),
                'sentence': sentence,
                'words': ' '.join(words),
                'tags': ' '.join(tags)
            })
        
        df = pd.DataFrame(readable_data)
        df.to_csv(output_path, index=False)
        print(f"Saved readable data to {output_path}")
        return df

def prepare_data():
    """Prepare all datasets"""
    preprocessor = POSDataPreprocessor()
    
    # Build vocabulary from training data
    print("Building vocabulary...")
    preprocessor.build_vocab('data/raw/Secs0-18 - training')
    
    # Save vocabulary
    preprocessor.save_vocab('data/models/vocab.pkl')
    
    # Baseado no vocabulário, podemos transformar os dados em sequências numéricas
    print("Processing training data...")
    train_X, train_y = preprocessor.text_to_sequences('data/raw/Secs0-18 - training')
    train_X_padded = preprocessor.pad_sequences(train_X)
    train_y_padded = preprocessor.pad_sequences(train_y)
    
    # Save readable training data
    #preprocessor.save_readable_data(train_X, train_y, 'data/models/train_readable.csv')
    
    # Prepare dev data
    print("Processing dev data...")
    dev_X, dev_y = preprocessor.text_to_sequences('data/raw/Secs19-21 - development')
    dev_X_padded = preprocessor.pad_sequences(dev_X)
    dev_y_padded = preprocessor.pad_sequences(dev_y)
    
    # Save readable dev data
    #preprocessor.save_readable_data(dev_X, dev_y, 'data/models/dev_readable.csv')
    
    # Prepare test data
    print("Processing test data...")
    test_X, test_y = preprocessor.text_to_sequences('data/raw/Secs22-24 - testing')
    test_X_padded = preprocessor.pad_sequences(test_X)
    test_y_padded = preprocessor.pad_sequences(test_y)
    
    # Save readable test data
    #preprocessor.save_readable_data(test_X, test_y, 'data/models/test_readable.csv')

    print(f"Dev shape: {dev_X_padded.shape}, {dev_y_padded.shape}")
    print(dev_X_padded[0])
    
    # Save processed data
    np.savez('data/models/pos_data.npz',
             train_X=train_X_padded, train_y=train_y_padded,
             dev_X=dev_X_padded, dev_y=dev_y_padded,
             test_X=test_X_padded, test_y=test_y_padded)
    
    print("Data preparation complete!")
    return preprocessor

if __name__ == "__main__":
    prepare_data()