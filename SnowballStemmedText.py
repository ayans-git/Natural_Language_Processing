from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

def SnowballStemmedText(text, language, ignoreStopwords):
    stemmer = SnowballStemmer(language, ignore_stopwords=ignoreStopwords)
    words = word_tokenize(text)
    stemmed_text = []
    for word in words:
        stemmed_text.append(stemmer.stem(word))
        stemmed_text.append(", ")
    return "".join(stemmed_text)
