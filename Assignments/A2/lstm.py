import torch
import torch.nn as nn
import re
import torch.nn.functional as F

embedding_dim = 128
hidden_dim = 256
num_layers = 2
dropout = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        x = self.embedding(x)  # Convert indices to embeddings
        x, hidden = self.lstm(x, hidden)  # Pass through LSTM
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)  # Project to vocabulary size
        return x, hidden  # (Batch, Seq_len, Vocab_size), hidden states


def predict_top_k_words_lstm(model, context_sentence, vocab, k, device='cpu'):
    """
    Predict the top-k next words for a given context sentence using the LSTM model.
    
    Args:
    - model: Trained LSTM model.
    - context_sentence: List of word indices representing the input sentence.
    - vocab: Dictionary mapping indices to words.
    - k: Number of top words to return.
    - device: 'cpu' or 'cuda'.
    
    Returns:
    - List of (word, probability) tuples.
    """
    model.eval()

    # Convert context sentence to tensor (batch_size=1, seq_len)
    context_tensor = torch.tensor(context_sentence, dtype=torch.long, device=device).unsqueeze(0)  # Shape: (1, seq_len)

    # Retrieve LSTM parameters
    num_layers = model.lstm.num_layers
    hidden_dim = model.lstm.hidden_size

    # Initialize hidden and cell states
    hidden = torch.zeros(num_layers, 1, hidden_dim, device=device)  # Shape: (num_layers, batch=1, hidden_dim)
    cell = torch.zeros(num_layers, 1, hidden_dim, device=device)

    with torch.no_grad():
        # Ensure input is correctly shaped for the model
        context_tensor = context_tensor.view(1, -1)  # Reshape to (batch=1, seq_len)
        
        # Pass through embedding layer manually if needed
        if hasattr(model, "embedding"):  
            context_tensor = model.embedding(context_tensor)  # Shape: (1, seq_len, embedding_dim)

        # Pass through LSTM
        output, _ = model.lstm(context_tensor, (hidden, cell))  

    # Get last word's output probabilities
    last_word_logits = output[:, -1, :]  # (1, vocab_size)
    probabilities = F.softmax(last_word_logits, dim=-1)  # Apply softmax to get probabilities

    # Get top-k words
    top_k_probs, top_k_indices = torch.topk(probabilities, k)  # (1, k)

    # Convert indices to words
    top_k_words = [(vocab[idx.item()], top_k_probs[0, i].item()) for i, idx in enumerate(top_k_indices[0])]

    return top_k_words


def sentence_to_tensor(sentence, word_to_idx, context_size, device='cpu'):
    """
    Convert a sentence into a tensor format suitable for the RNN model.
    
    Args:
        sentence (str): Input sentence.
        word_to_idx (dict): Mapping from words to indices.
        context_size (int): Number of words used as context.
        device (str): Device to place the tensor ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Context tensor of shape (1, context_size).
    """
    sentence = sentence.lower()
    words = re.sub(r"[^a-zA-Z\s]", "", sentence).split()
    
    if len(words) < context_size:
        raise ValueError(f"Input sentence must have at least {context_size} words")
    
    # Extract the last `context_size` words
    context_words = words[-context_size:]
    
    # Convert words to indices
    context_indices = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in context_words]
    
    # Convert to tensor (1, context_size)
    context_tensor = torch.tensor([context_indices], dtype=torch.long).to(device)
    
    return context_tensor

def lstm_pred(model_path,context_sentence,k):
    checkpoint = torch.load(model_path, map_location="cpu")

    # Restore vocabulary and prevent dynamic resizing
    word_to_idx = checkpoint["word_to_idx"]
    idx_to_word = checkpoint["idx_to_word"]

    # Use stored vocab size
    vocab_size = checkpoint["hyperparams"]["vocab_size"]

    # Extract hyperparameters
    embedding_dim = checkpoint["hyperparams"]["embedding_dim"]
    hidden_dim = checkpoint["hyperparams"]["hidden_dim"]
    num_layers = checkpoint["hyperparams"]["num_layers"]

    # Build model with correct vocab size
    model_lstm_loaded = LSTMLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers)

    # Load model weights
    model_lstm_loaded.load_state_dict(checkpoint["model_state_dict"])
    model_lstm_loaded.eval()

    context_tensor = sentence_to_tensor(context_sentence, word_to_idx=word_to_idx, context_size=3, device='cpu')

    predictions = predict_top_k_words_lstm(model_lstm_loaded, context_tensor.tolist()[0], idx_to_word, k=5, device='cpu')

    return predictions

