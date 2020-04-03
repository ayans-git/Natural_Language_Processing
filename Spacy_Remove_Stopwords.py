import spacy
spacy_corpus = spacy.load('en_core_web_sm')

def remove_stopwords(corpus, custom_stopwords):
    spacy_english_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    spacy_document = spacy_corpus(corpus)
    for w in custom_stopwords:
        spacy_corpus.vocab[w].is_stop = True
    words = [word.text for word in spacy_document if not word.is_stop]
    return words
