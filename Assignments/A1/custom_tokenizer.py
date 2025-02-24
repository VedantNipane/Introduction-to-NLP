import re

def custom_tokenizer(s):
    # Step 1: Normalize case
    s = s.lower()

    s = re.sub(r"localhost:\d{4}\/?(?:[\w/\-?=%.]+)?|http:\/\/?localhost:\d{4}\/?(?:[\w/\-?=%.]+)?|(?:(?:https?|ftp|localhost):\/\/)?[\w/\-?=%]+\.[\w/\-&?=%]+", "<URL>", s)
    s = re.sub(r"((?<![\S])#\w+)", "<HASHTAG>", s)
    s = re.sub(r"((?<![\S])@\w+)", "<MENTION>", s)
    s = re.sub(r'\b\d+(?:\.\d+)?\b', '<NUM>', s)
    s = re.sub(r'[$€£¥]', '<CURR>', s)
    s = re.sub(r'\b\d{1,2}(?::\d{2})?\s*[APap][Mm]\b', '<TIME>', s)
    s = re.sub(r'\b\d+(?:\.\d+)?%\b', '<PERCENTAGE>', s)
    s = re.sub(r'\n|\t', ' ', s)

    sentences = re.split(r'(?<=[.!?])\s+', s)

    tokenized_sentences = []
    for sentence in sentences:
        # Tokenize words and preserve punctuation
        tokens = re.findall(r'\b\w+\b|[.,!?]', sentence)
        
        # Capitalize first word of each sentence
        if tokens:
            tokens[0] = tokens[0].capitalize()
        
        tokenized_sentences.append(tokens)

    return tokenized_sentences

if __name__ == "__main__":
    text = input("Your Text: ")
    tokenized_text = custom_tokenizer(text)
    print("Tokenized Text: ", tokenized_text)
