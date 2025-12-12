import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Data Loading ---
def load_data():
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("tweet_eval", "sentiment")
    
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    val_df = pd.DataFrame(dataset['validation'])

    return {
        "train": (train_df['text'].tolist(), train_df['label'].tolist()),
        "test": (test_df['text'].tolist(), test_df['label'].tolist()),
        "val": (val_df['text'].tolist(), val_df['label'].tolist()),
        "raw_dataset": dataset # Keep original object for HF Trainer
    }

def build_vocab(text_list):
    """Creates a simple vocabulary dictionary from text."""
    vocab = {word: i+1 for i, word in enumerate(set(" ".join(text_list).split()))}
    return vocab

def encode_text_lstm(text_list, tokenizer, max_len=50):
    """Manually encodes text into padded sequences for LSTM."""
    encoded = []
    for text in text_list:
        tokens = [tokenizer.get(w, 0) for w in text.split()]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens)) # Padding
        else:
            tokens = tokens[:max_len]
        encoded.append(tokens)
    return torch.tensor(encoded, dtype=torch.long)

def compute_metrics(eval_pred):
    """
    Metric computation function for HuggingFace Trainer.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }