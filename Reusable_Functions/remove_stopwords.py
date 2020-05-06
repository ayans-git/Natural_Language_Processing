#nltk.download('stopwords')
#nltk.download('punkt')

import nltk

def remove_stopwords(corpus, language):
    language_stopwords = nltk.corpus.stopwords.words(language)
    words = nltk.tokenize.word_tokenize(corpus)
    words = [word for word in words if not word in language_stopwords]
    return words
