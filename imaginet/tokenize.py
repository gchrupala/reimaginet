# Code adapted from https://github.com/IndicoDataSolutions/Passage
# Copyright (c) 2015 IndicoDataSolutions
def tokenize(text):
    punctuation = set(string.punctuation)
    punctuation.add('\n')
    punctuation.add('\t')
    punctuation.add('')
    tokenized = []
    w = ''
    for t in text:
        if t in punctuation:
            tokenized.append(w)
            tokenized.append(t)
            w = ''
        elif t == ' ':
            tokenized.append(w)
            w = ''
        else:
            w += t
    if w != '':
        tokenized.append(w)
    tokenized = [token for token in tokenized if token]
    return tokenized
