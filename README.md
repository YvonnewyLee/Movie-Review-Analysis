# Movie Review Sentiment Analysis

This project focuses on building a sentiment analysis model for classifying movie reviews as positive or negative. Initially, a recurrent neural network (RNN) using LSTM was used to process the IMDB Movie Review Dataset. In the second part, transfer learning is applied by fine-tuning a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model to improve performance.

## Project Overview

This project performs sentiment analysis on IMDB movie reviews by building two models:
- **Part A**: An LSTM-based model to classify reviews as positive or negative.
- **Part B**: A fine-tuned BERT model using transfer learning to improve classification accuracy.

## Dataset

The project uses the **IMDB Movie Review Dataset**, consisting of 50,000 labeled reviews (25,000 positive and 25,000 negative) for training and evaluation.

## Data Preprocessing

Data preprocessing steps include:
- **Tokenization**: The reviews are tokenized into words using either the BERT tokenizer or custom tokenization for the LSTM model.
- **Padding**: All sequences are padded to the same length for efficient model training.
- **Attention Masks**: For BERT, attention masks are generated to inform the model about padding tokens.

## Model Architecture

### Part A: LSTM Model
The model is based on LSTM, which is suitable for handling sequential data like text. The LSTM layers extract features from the text sequences, and a fully connected layer produces the final sentiment classification.

```python
class SentimentRNN(nn.Module):
    def __init__(self, input_shape=(500, 1), num_classes=2, dim_hidden=20, dropout_rate=0, network_type="LSTM"):
        super(SentimentRNN, self).__init__()
        # LSTM layers
        # Fully connected output layer
```

### Part B: BERT Model
In Part B, we use a pre-trained BERT model and fine-tune it on the movie review dataset. We use the BERT tokenizer to preprocess the text and input it into the model, which produces embeddings. The embeddings are then passed through a classification head.

#### Using Pooled Output:
```python
class SentimentClassifierPooled(nn.Module):
    def __init__(self, n_classes=2):
        super(SentimentClassifierPooled, self).__init__()
        self.bert = bert_model
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(768, n_classes)
```

#### Using Last Hidden State:
```python
class SentimentClassifierLast(nn.Module):
    def __init__(self, n_classes=2):
        super(SentimentClassifierLast, self).__init__()
        self.bert = bert_model
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(768, n_classes)
```

## Training

### Part A: Training the LSTM Model
The LSTM model is trained using:
- **Loss Function**: Cross-entropy loss.
- **Optimizer**: Adam optimizer.
- **Batch Size**: 16.
- **Epochs**: 80 epochs.

### Part B: Training the BERT Model
BERT fine-tuning is performed using:
- **Learning Rate**: A low learning rate (1e-5) to ensure careful fine-tuning.
- **Optimizer**: AdamW optimizer (recommended for BERT).
- **Epochs**: Typically 5-10 epochs for fine-tuning.

```python
def train_bert(model, train_loader, valid_loader, num_epoch=5, learning_rate=1e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop with loss and accuracy tracking
```

## Evaluation

The model performance is evaluated using:
- **Accuracy**: Calculated on the training, validation, and test sets.
- **False Positive/Negative Rates**: Important for understanding misclassifications.
- **Misclassified Reviews**: Analysis of specific reviews misclassified by the models.

## Hyperparameter Tuning

We experimented with several hyperparameters to optimize the performance:
1. **Dropout Rate**: Increased for better generalization.
2. **Learning Rate**: Fine-tuned to achieve the best result without overfitting.
3. **Epochs**: Experimented with different epoch counts to avoid overfitting.
4. **Model Structure**: Tested both pooled output and last hidden state from BERT to compare results.



