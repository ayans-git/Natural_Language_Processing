from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

def PorterStemmedText(text):
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    stemmed_text = []
    for word in words:
        stemmed_text.append(stemmer.stem(word))
        stemmed_text.append(", ")
    return "".join(stemmed_text)
