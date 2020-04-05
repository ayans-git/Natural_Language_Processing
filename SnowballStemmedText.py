from nltk.stem.snowball import SnowballStemmer

def SnowballStemmedText(text, language, ignoreStopwords):
    stemmer = SnowballStemmer(language, ignore_stopwords=ignoreStopwords)
    return stemmer.stem(text)
