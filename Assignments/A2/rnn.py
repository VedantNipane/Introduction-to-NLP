import torch.nn as nn
import re
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """
        Initialize the Vanilla RNN model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of word embeddings
            hidden_dim (int): Dimension of hidden state
        """
        super(VanillaRNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
            hidden (torch.Tensor, optional): Initial hidden state
            
        Returns:
            tuple: (output probabilities, final hidden state)
        """
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded, hidden)
        output = self.fc(rnn_out)  # Shape: (batch_size, sequence_length, vocab_size)
        
        return output, hidden
    
    def predict_next_word(self, context, idx_to_word, top_k=5):
        """
        Predict the next word given a context.
        
        Args:
            context (torch.Tensor): Input context tensor
            idx_to_word (dict): Mapping from indices to words
            top_k (int): Number of top predictions to return
            
        Returns:
            list: Top k predicted words with their probabilities
        """
        self.eval()
        with torch.no_grad():
            output, _ = self(context)
            probabilities = torch.softmax(output[:, -1], dim=-1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            predictions = [
                (idx_to_word[idx.item()], prob.item())
                for idx, prob in zip(top_indices[0], top_probs[0])
            ]
        
        return predictions


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
    context_indices = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in context_words]
    context_tensor = torch.tensor([context_indices], dtype=torch.long).to(device)
    
    return context_tensor



def rnn_pred(model_path,context_sentence,k):
    # Load checkpoint
    checkpoint = torch.load(model_path)

    # Extract saved components
    word_to_idx = checkpoint["word_to_idx"]
    idx_to_word = checkpoint["idx_to_word"]
    hyperparams = checkpoint["hyperparams"]
    embedding_dim = hyperparams["embedding_dim"]
    hidden_dim = hyperparams["hidden_dim"]
    vocab_size = hyperparams["vocab_size"]
    model_rnn_1 = VanillaRNN(vocab_size, embedding_dim, hidden_dim)
    model_rnn_1.load_state_dict(checkpoint["model_state_dict"])
    model_rnn_1.eval()
    context_tensor = sentence_to_tensor(context_sentence, word_to_idx=word_to_idx, context_size=3, device='cpu')
    # Predict next word using your RNN model
    predictions = model_rnn_1.predict_next_word(context_tensor, idx_to_word, top_k=k)
    return predictions
    