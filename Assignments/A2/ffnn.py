import torch.nn as nn
import re
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FFNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_size, embedding_dim=100, hidden_dim=256):
        super(FFNNLanguageModel, self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Flatten the embeddings and feed through fully connected layers
        self.ff_layers = nn.Sequential(
            nn.Linear(context_size * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # x shape: (batch_size, context_size)
        embeds = self.embeddings(x)  # (batch_size, context_size, embedding_dim)
        
        # Flatten the embeddings
        batch_size = embeds.shape[0]
        flattened = embeds.view(batch_size, -1)
        
        # Feed through layers
        hidden = self.ff_layers(flattened)
        log_probs = self.log_softmax(hidden)
        return log_probs

def predict_next_word(model, sentence, word_to_idx, idx_to_word, context_size=3, device='cpu', top_k=5):
    sentence = sentence.lower()
    words = re.sub(r"[^a-zA-Z\s]", "", sentence).split()
    if len(words) < context_size:
        raise ValueError(f"Input sentence must have at least {context_size} words")
    
    context_words = words[-context_size:]
    context_indices = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in context_words]
    context_tensor = torch.tensor([context_indices], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        log_probs = model(context_tensor)
        probs = torch.exp(log_probs)
        top_k_probs, top_k_indices = torch.topk(probs[0], k=top_k)
        
        predictions = [(idx_to_word[idx.item()], prob.item()) 
                      for idx, prob in zip(top_k_indices, top_k_probs)]
    
    return predictions


def ffnn_pred(model_path,context_sentence,k,n):
    ## Model Loading
    checkpoint = torch.load(model_path)
    # Load the saved vocabulary
    word_to_idx = checkpoint["word_to_idx"]
    vocab_size = checkpoint["vocab_size"]
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    # Ensure vocab size is the same before initializing the model
    model = FFNNLanguageModel(vocab_size=vocab_size, context_size=n, embedding_dim=100, hidden_dim=256)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    predictions = predict_next_word(
        model = model,
        sentence=context_sentence,
        word_to_idx=word_to_idx,
        idx_to_word=idx_to_word,
        context_size=n,
        top_k=k
    )

    return predictions
    