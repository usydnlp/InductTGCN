import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import nltk
import re


def encode_labels(original_labels_train, original_labels_test, args):

    unique_labels=np.unique(original_labels_train + original_labels_test)

    num_class = len(unique_labels)
    lEnc = LabelEncoder()
    lEnc.fit(unique_labels)
    if not args.easy_copy:
        print("Labels:", unique_labels)
        print("Encoded:", lEnc.transform(unique_labels))

    train_labels = lEnc.transform(original_labels_train)
    test_labels = lEnc.transform(original_labels_test)

    labels = train_labels.tolist()+test_labels.tolist()

    return labels,num_class


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def preprocess_data(original_train_sentences, original_test_sentences, args):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    original_word_freq = {}  # to remove rare words
    for sentence in original_train_sentences:
        temp = clean_str(sentence)
        word_list = temp.split()
        for word in word_list:
            if word in original_word_freq:
                original_word_freq[word] += 1
            else:
                original_word_freq[word] = 1   

    sentences = original_train_sentences + original_test_sentences
    tokenize_sentences = []
    word_list_dict = {}
    for sentence in sentences:
        temp = clean_str(sentence)
        word_list_temp = temp.split()
        doc_words = []
        for word in word_list_temp:
            if word in original_word_freq and word not in stop_words and original_word_freq[word] >= args.remove_limit:
                doc_words.append(word)
                word_list_dict[word] = 1
        tokenize_sentences.append(doc_words)
    word_list = list(word_list_dict.keys())

    len_list = [len(l) for l in tokenize_sentences[:len(original_train_sentences)]]
    if not args.easy_copy:
        print("Average Length:", sum(len_list)/len(len_list))
        

    return tokenize_sentences, word_list






    