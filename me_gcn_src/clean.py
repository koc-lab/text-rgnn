import re

import nltk
from nltk.corpus import stopwords


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def tokenize_and_clean(docs):
    original_word_freq = {}
    for doc in docs:
        temp = clean_str(doc)
        word_list = temp.split()
        for word in word_list:
            if word in original_word_freq:
                original_word_freq[word] += 1
            else:
                original_word_freq[word] = 1

    tokenized_docs = []
    word_list_dict = {}

    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

    for doc in docs:
        temp = clean_str(doc)
        word_list_temp = temp.split()
        doc_words = []
        for word in word_list_temp:
            if word not in stop_words and original_word_freq[word] >= 5:
                doc_words.append(word)
                word_list_dict[word] = 1
        tokenized_docs.append(doc_words)
    word_list = list(word_list_dict.keys())
    vocab_length = len(word_list)

    word_id_map = {}
    for i in range(vocab_length):
        word_id_map[word_list[i]] = i

    return word_list, tokenized_docs, word_id_map
