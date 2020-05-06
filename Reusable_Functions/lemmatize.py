from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def lemmatize(text, punctuations):
    words = word_tokenize(text)
    lemma_text = []
    
    for word in words:
        if word in punctuations:
            words.remove(word)
    
    for word in words:
        lemma_text.append(WordNetLemmatizer().lemmatize(word, pos='v'))
        
    return lemma_text
