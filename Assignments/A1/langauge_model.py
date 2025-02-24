import re
import numpy as np
import math
from collections import Counter
from scipy.stats import linregress
from custom_tokenizer import custom_tokenizer 
from n_gram_model import n_gram
import os
import sys

class LanguageModel:
    def __init__(self, n, smoothing_type, corpus_text):
        """
        Initialize Language Model
        
        Args:
        - n (int): N-gram size (1, 2, 3, etc.)
        - smoothing_type (str): 
            'l' - Laplace 
            'g' - Good-Turing
            'i' - Linear Interpolation
        - corpus_text (str): Raw text corpus
        """
        if not corpus_text or len(corpus_text.strip()) == 0:
            raise ValueError("Corpus text cannot be empty")
        
        self.n = n
        self.smoothing_type = smoothing_type
        self.corpus_text = corpus_text
        self.tokenized_sentences = custom_tokenizer(corpus_text)
        
        # Validate tokenization
        if not self.tokenized_sentences:
            raise ValueError("No sentences could be tokenized from the corpus")
        
        self.ngram_model = n_gram(n, self.tokenized_sentences)
        self.probabilities = {}
        self.vocab = set()
        
    def train(self):
        """Train the language model using the specified smoothing technique"""
        # Collect vocabulary
        for sentence in self.tokenized_sentences:
            self.vocab.update(sentence)
        
        # Validate n-gram model
        if not self.ngram_model:
            raise ValueError(f"Could not generate {self.n}-gram model")
        
        # Apply selected smoothing technique
        if self.smoothing_type == 'l':
            self._laplace_smoothing()
        elif self.smoothing_type == 'g':
            self._good_turing_smoothing()
        elif self.smoothing_type == 'i':
            self._linear_interpolation()
        
        # Validate probabilities
        if not self.probabilities:
            raise ValueError("No probabilities could be generated")
        
    
    def _laplace_smoothing(self, alpha=1):
        """Apply Laplace (Add-One) Smoothing or Additive Smoothing"""
        vocab_size = len(self.vocab)# total unique vocabulary size
        
        for prefix, next_tokens in self.ngram_model.items():
            total_context_count = sum(next_tokens.values())

            for token, count in next_tokens.items():
                smoothed_prob = (count + alpha) / (total_context_count + alpha * vocab_size)
                full_ngram = prefix + (token,)
                self.probabilities[full_ngram] = smoothed_prob

    
    def _good_turing_smoothing(self):
        """Apply Good-Turing Smoothing"""
        # Flatten n-gram counts
        all_counts = []
        for next_tokens in self.ngram_model.values():
            all_counts.extend(next_tokens.values())
        
        # Compute frequency of frequencies
        freq_freq = Counter(all_counts)
        sorted_freq = sorted(freq_freq.items())
        
        # Compute Z values, expected number of unseen events
        Z = []
        for i in range(len(sorted_freq)):
            try:
                if i == 0 and len(sorted_freq) > 1:
                    Z.append(sorted_freq[0][0] / (2 * sorted_freq[1][0]))
                elif i == len(sorted_freq) - 1 and len(sorted_freq) > 1:
                    Z.append(2 * sorted_freq[-1][0] / (sorted_freq[-1][0] - sorted_freq[-2][0]))
                elif len(sorted_freq) > 2:
                    Z.append(2 * sorted_freq[i][0] / (sorted_freq[i+1][0] - sorted_freq[i-1][0]))
            except (IndexError, ZeroDivisionError):
                Z.append(0)
        
        # Ensure non-empty Z and log values
        Z = [max(z, 1e-10) for z in Z]
        log_freq = np.log([max(f, 1e-10) for f, _ in sorted_freq])
        log_Z = np.log(Z)
        
        # Linear regression
        try:
            slope, intercept, _, _, _ = linregress(log_freq, log_Z)
        except Exception:
            # Fallback to default probabilities if regression fails
            slope, intercept = 0, 0
        
        # Compute adjusted probabilities
        for prefix, next_tokens in self.ngram_model.items():
            total_context_count = sum(next_tokens.values())
            for token, count in next_tokens.items():
                try:
                    adjusted_count = (count + 1) * np.exp(intercept + slope * np.log(max(count, 1))) / \
                                    np.exp(intercept + slope * np.log(max(count - 1, 1)))
                except Exception:
                    adjusted_count = count + 1
                
                smoothed_prob = adjusted_count / total_context_count
                full_ngram = prefix + (token,)
                self.probabilities[full_ngram] = max(smoothed_prob, 1e-10)

    def _linear_interpolation(self):
        """Linear Interpolation across different order models"""
        for prefix, next_tokens in self.ngram_model.items():
            total_context_count = sum(next_tokens.values())
            for token, count in next_tokens.items():
                # Simple interpolation: context probability / total context count
                prob = count / total_context_count
                full_ngram = prefix + (token,)
                self.probabilities[full_ngram] = prob
    
    def _calculate_order_probability(self, order, ngram):
        """Calculate probability for a specific n-gram order"""
        if order == 1:
            # Unigram probability
            return self.ngram_counts[1].get(ngram[-1:], 0) / sum(self.ngram_counts[1].values())
        
        # Higher order probabilities
        context = ngram[:-1]
        
        # Count of n-gram
        ngram_count = self.ngram_counts[order].get(ngram, 0)
        
        # Count of context
        context_count = sum(count for gram, count in self.ngram_counts[order].items() if gram[:-1] == context)
        
        return ngram_count / context_count if context_count > 0 else 0

    def _estimate_lambda_weights(self):
        """Estimate lambda weights for each n-gram order"""
        lambda_weights = {}
        total_weight = 0
        
        for order in range(1, self.n + 1):
            # Simple heuristic: weight based on count of n-grams
            order_count = sum(self.ngram_counts[order].values())
            lambda_weights[order] = order_count
            total_weight += order_count
        
        # Normalize weights
        for order in lambda_weights:
            lambda_weights[order] /= total_weight
        
        return lambda_weights
    
    def _linear_interpolation(self):
        # Collect n-gram counts for all orders from 1 to n
        self.ngram_counts = {}
        
        for order in range(1, self.n + 1):
            self.ngram_counts[order] = {}
            for sentence in self.tokenized_sentences:
                # Add start tokens for context
                padded_sentence = ["<start>"] * (order - 1) + sentence + ["<end>"]
                
                # Count n-grams of current order
                ngrams = [tuple(padded_sentence[i:i+order]) for i in range(len(padded_sentence)-order+1)]
                for gram in ngrams:
                    self.ngram_counts[order][gram] = self.ngram_counts[order].get(gram, 0) + 1
        
        # Estimate lambda weights
        self.lambda_weights = self._estimate_lambda_weights()
        
        # Compute interpolated probabilities
        for prefix, next_tokens in self.ngram_model.items():
            for token, count in next_tokens.items():
                # Calculate probabilities for different n-gram orders
                prob = 0
                full_ngram = prefix + (token,)
                
                for order in range(1, self.n + 1):
                    # Calculate probability for each n-gram order
                    order_prob = self._calculate_order_probability(order, full_ngram)
                    prob += self.lambda_weights.get(order, 0) * order_prob
                
                self.probabilities[full_ngram] = prob

    
    def perplexity(self, test_sentences):
        """Calculate perplexity of test sentences"""
        if not test_sentences:
            raise ValueError("Test sentences cannot be empty")
        
        log_prob_sum = 0
        total_ngrams = 0
        
        for sentence in test_sentences:
            padded_sentence = ["<start>"] * (self.n - 1) + sentence + ["<end>"]
            ngrams = [tuple(padded_sentence[i:i+self.n]) for i in range(len(padded_sentence)-self.n+1)]
            
            for ngram in ngrams:
                prob = self.probabilities.get(ngram, 1e-10)
                log_prob_sum += math.log(prob)
                total_ngrams += 1
        
        # Prevent division by zero
        if total_ngrams == 0:
            return float('inf')
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Total n-gram probabilities: {len(self.probabilities)}")

        return math.exp(-log_prob_sum / total_ngrams)

def main():
    # Check correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python3 language_model.py <lm_type> <corpus_path>")
        print("LM types: l (Laplace), g (Good-Turing), i (Interpolation)")
        sys.exit(1)
    
    # Parse arguments
    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    
    # Validate language model type
    if lm_type not in ['l', 'g', 'i']:
        print("Invalid language model type. Use 'l', 'g', or 'i'.")
        sys.exit(1)
    
    # Read corpus file
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_text = f.read()
    except FileNotFoundError:
        print(f"Corpus file not found: {corpus_path}")
        sys.exit(1)
    
    # Create and train language model (default to 3-gram)
    try:
        lm = LanguageModel(n=3, smoothing_type=lm_type, corpus_text=corpus_text)
        lm.train()
    except Exception as e:
        print(f"Error initializing language model: {e}")
        sys.exit(1)
    
    # Interactive input loop
    while True:
        try:
            # Prompt for input sentence
            sentence = input("input sentence: ")
            if(sentence=='exit'):
                exit()
            # Tokenize input sentence
            tokenized_sentence = custom_tokenizer(sentence)[0]
            
            # Add start and end tokens
            padded_sentence = ["<start>"] * 2 + tokenized_sentence + ["<end>"]
            
            # Calculate sentence probability
            sentence_prob = 1.0
            for i in range(2, len(padded_sentence)):
                ngram = tuple(padded_sentence[i-2:i+1])
                prob = lm.probabilities.get(ngram, 1e-10)
                sentence_prob *= prob
            
            # Print probability with 8 decimal places
            print(f"score: {sentence_prob:.8f}")
        
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error processing sentence: {e}")

if __name__ == "__main__":
    main()