# Social Media Sentiment Analysis: From Traditional ML to Transformers

## 📌 Project Overview
This project explores the evolution of Natural Language Processing (NLP) techniques for social media sentiment analysis. We implement and compare three distinct modeling paradigms to classify tweets into **Negative**, **Neutral**, or **Positive** sentiments:

1.  **Baseline**: Traditional Machine Learning (Logistic Regression with TF-IDF).
2.  **Deep Learning**: Recurrent Neural Networks (LSTM with Word Embeddings).
3.  **State-of-the-Art**: Transformer-based Transfer Learning (Fine-tuned DistilBERT).

[cite_start]**Goal**: To demonstrate the superior performance of Transformer models in handling context, slang, and noise inherent in social media text[cite: 1, 7].

# Dataset Information

The dataset used in this project is **TweetEval (Sentiment Benchmark)**.

## Access
The data is **not stored locally** in this repository to save space. Instead, it is automatically downloaded via the Hugging Face `datasets` library when running the training script.

## Source
* **Name**: TweetEval
* **Task**: Sentiment Analysis
* **Link**: [https://huggingface.co/datasets/tweet_eval](https://huggingface.co/datasets/tweet_eval)

## Data Structure
The dataset consists of tweets labeled into three sentiment classes:
* `0`: Negative
* `1`: Neutral
* `2`: Positive

## 📂 Repository Structure
```text
.
├── checkpoints/                # Directory for model weights
│   └── distilbert_sentiment/   # Place downloaded DistilBERT files here
├── demo/
│   └── demo.ipynb              # Interactive demo script
├── results/                    # Generated predictions and plots
├── src/                        # Source code for training and evaluation
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies

# Model Setup (Important)
Since the trained Transformer model files are large, they are not hosted directly on GitHub.

Download the model files from this Google Drive link:

[https://drive.google.com/drive/folders/1fepelZy0h7GUctGvYYXY-6qZJZ3OjcNd?usp=sharing]

Unzip/Place the files inside the checkpoints/distilbert_sentiment/ folder.

Ensure the folder contains config.json, model.safetensors, vocab.txt, etc.

# How to Run
Run the Demo
Navigate to the demo/ folder.

Open demo.ipynb using Jupyter Notebook or Lab.

Run all cells to load the model and predict sentiment on sample texts.

Results will be saved to results/demo_predictions.csv.

Reproduce Training
To retrain the models from scratch, run the main script:
python src/main.py

Experiment Results

Logistic Regression: ~59.6% Accuracy 


LSTM: ~44.0% Accuracy 


DistilBERT: ~69.4% Accuracy 

Acknowledgments
Dataset: TweetEval (Sentiment) from HuggingFace.

Libraries: PyTorch, Scikit-learn, Transformers.