import sys
import heapq
from custom_tokenizer import custom_tokenizer
from n_gram_model import n_gram
from langauge_model import LanguageModel
def next_word_predictions(language_model, sentence, top_k=5):
    """
    Predicts the next word for a given sentence using the trained language model.

    Parameters:
        language_model (LanguageModel): The trained language model.
        sentence (str): The input sentence for prediction.
        top_k (int): The number of top predictions to return.

    Returns:
        list: A list of tuples of the top_k predicted words with their probabilities.
    """
    # Tokenize and process the input sentence
    tokens = custom_tokenizer(sentence.lower())[0]  # Ensure consistency
    tokens = ["<start>"] * (language_model.n - 1) + tokens  # Add start tokens for context

    # Extract the prefix from the last (N-1) tokens
    prefix = tuple(tokens[-(language_model.n - 1):])

    # Find possible next words and their probabilities
    predictions = {}
    for full_ngram, prob in language_model.probabilities.items():
        # Check if the prefix matches the context of the n-gram
        if full_ngram[:-1] == prefix:
            predictions[full_ngram[-1]] = prob

    # Get the top_k predictions sorted by probability
    top_predictions = heapq.nlargest(top_k, predictions.items(), key=lambda x: x[1])

    return top_predictions

def main():
    # Check correct number of arguments
    if len(sys.argv) != 4:
        print("Usage: python3 generator.py <lm_type> <corpus_path> <k>")
        print("LM types: l (Laplace), g (Good-Turing), i (Interpolation)")
        sys.exit(1)
    
    # Parse arguments
    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    top_k = int(sys.argv[3])
    
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
        lm = LanguageModel(n=5, smoothing_type=lm_type, corpus_text=corpus_text)
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
            # Get top k predictions
            predictions = next_word_predictions(lm, sentence, top_k)
            
            # Print predictions
            for word, prob in predictions:
                print(f"{word} {prob:.2f}")
        
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error processing sentence: {e}")

if __name__ == "__main__":
    main()