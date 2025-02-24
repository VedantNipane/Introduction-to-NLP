import os
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import warnings
import sys

## Custom Imports
import ffnn
import rnn
import lstm
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text1_path = 'Pride and Prejudice - Jane Austen.txt'
text2_path = 'Ulysses - James Joyce.txt'

def model_path_finder(lm_type, corpus_path, n=None):
    corpus_suffix = 'corpus1' if corpus_path == 'Pride and Prejudice - Jane Austen.txt' else 'corpus2'
    
    suffix = ''
    if lm_type == 'f':
        suffix = f'ffnn_{corpus_suffix}_N{n}' if n is not None else f'ffnn_{corpus_suffix}'
    elif lm_type == 'r':  # Use elif instead of "else if"
        suffix = f'rnn_{corpus_suffix}'
    else:
        suffix = f'lstm_{corpus_suffix}'

    return f'Models/2021102040_{suffix}.pth'

def print_predictions(sentence, predictions):
    print(f"\nInput sentence: {sentence}")
    print("Top 5 predicted next words:")
    for word, prob in predictions:
        print(f"{word}: {prob:.4f}")

def main():
    # Check correct number of arguments
    if len(sys.argv) != 4:
        print("Usage: python3 generator.py <lm_type> <corpus_path> <k>")
        print("LM types: f (Feedforward Neural Network), r (Recurrent Neural Network), l (Long Short-Term Memory)")
        sys.exit(1)
    
    # Parse arguments
    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    top_k = int(sys.argv[3])
    
    # Validate language model type
    if lm_type not in ['f', 'r', 'l']:
        print("Invalid language model type. Use 'f', 'r', or 'l'.")
        sys.exit(1)

    # Validate Corpus
    if corpus_path not in ['Pride and Prejudice - Jane Austen.txt','Ulysses - James Joyce.txt']:
        print("Invalid corpus")
        sys.exit(1)
    
    n = None
    if(lm_type=='f'):
        n = int(input("Choose the N grams (3 or 5): "))
    
    model_path = model_path_finder(lm_type,corpus_path,n)
    context_sentence  = input("Input Sentence: ")
    
    predictions = None
    if(lm_type=='f'):
        predictions = ffnn.ffnn_pred(model_path,context_sentence,top_k,n)
    elif(lm_type=='r'):
        predictions = rnn.rnn_pred(model_path,context_sentence,top_k)
    else:
        predictions = lstm.lstm_pred(model_path,context_sentence,top_k)

    print_predictions(context_sentence,predictions)
    


if __name__ == "__main__":
    main()