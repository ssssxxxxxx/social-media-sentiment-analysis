import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import joblib

# Import from our own modules
from model import SimpleLSTM, get_distilbert_model
from utils import load_data, build_vocab, encode_text_lstm, compute_metrics

# For Logistic Regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# For DistilBERT
from transformers import DistilBertTokenizer, Trainer, TrainingArguments

def main():
    # 0. Setup Directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. Load Data
    data = load_data()
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]
    X_val, y_val = data["val"]

    print("\n--- Training Model 1: Logistic Regression ---")
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred_lr = lr_model.predict(X_test_tfidf)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")

    # Save
    joblib.dump(lr_model, 'checkpoints/lr_model.pkl')
    joblib.dump(vectorizer, 'checkpoints/tfidf_vectorizer.pkl')

    print("\n--- Training Model 2: LSTM ---")
    # Prepare Data
    vocab = build_vocab(X_train)
    X_train_tensor = encode_text_lstm(X_train, vocab)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = encode_text_lstm(X_test, vocab)
    
    # Init Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_model = SimpleLSTM(vocab_size=len(vocab)+1).to(device)
    
    # Train Loop (Simplified)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    lstm_model.train()
    for epoch in range(2): # Keep epochs low for demo purposes, increase for real result
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            output = lstm_model(texts)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    
    print("LSTM training finished.")
    
    # Save Model & Vocab
    torch.save(lstm_model, 'checkpoints/lstm_model.pth')
    with open('checkpoints/lstm_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    print("\n--- Training Model 3: DistilBERT ---")
    os.environ["WANDB_DISABLED"] = "true"
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    
    # Process dataset object
    hf_dataset = data["raw_dataset"]
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
    
    bert_model = get_distilbert_model()
    
    training_args = TrainingArguments(
        output_dir='./results_bert',
        num_train_epochs=1, # Adjust as needed
        per_device_train_batch_size=16,
        logging_steps=100,
        save_strategy="no" # We save manually at end to keep it simple
    )
    
    trainer = Trainer(
        model=bert_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # Evaluate
    results = trainer.predict(tokenized_datasets["test"])
    print(f"DistilBERT Accuracy: {results.metrics['test_accuracy']:.4f}")

    # Save
    bert_model.save_pretrained("checkpoints/distilbert_sentiment")
    tokenizer.save_pretrained("checkpoints/distilbert_sentiment")
    print("All models saved successfully!")

if __name__ == "__main__":
    main()