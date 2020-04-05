from nltk.stem.porter import PorterStemmer

def PorterStemmedText(text):
    stemmer = PorterStemmer()
    return stemmer.stem(text)
