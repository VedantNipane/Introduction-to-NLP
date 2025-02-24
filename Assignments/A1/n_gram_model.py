from custom_tokenizer import custom_tokenizer

def n_gram(N, tokenized_sentences):
    """
    Generates an N-sized N-gram model from the given corpus with start and end tokens.

    Parameters:
        N (int): The size of the N-grams to generate.
        tokenized_sentences (str): Tokenized Sentences Processed by Custom Tokenizer

    Returns:
        dict: A nested dictionary representing the N-gram model.
              Keys: Prefix tuples (N-1 tokens).
              Values: Dictionary of the next token and its frequency.
    """
    ngram_model = {}

    for tokens in tokenized_sentences:
        # Add start and end tokens
        tokens = ["<start>"] * (N - 1) + tokens + ["<end>"]

        # Create N-grams
        ngrams = [tuple(tokens[i:i + N]) for i in range(len(tokens) - N + 1)]

        # Populate the N-gram model
        for gram in ngrams:
            prefix = gram[:-1]  # First N-1 tokens
            next_token = gram[-1]  # Last token

            if prefix not in ngram_model:
                ngram_model[prefix] = {next_token: 1}
            else:
                ngram_model[prefix][next_token] = ngram_model[prefix].get(next_token, 0) + 1

    return ngram_model

