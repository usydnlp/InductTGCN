# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from tqdm.auto import tqdm
import scipy.sparse as sp
from math import log
import numpy as np


def ordered_word_pair(a, b):
    if a > b:
        return b, a
    else:
        return a, b

def get_adj(tokenize_sentences,train_size,word_id_map,word_list,args):
    window_size = 20
    total_W = 0
    word_occurrence = {}
    word_pair_occurrence = {}

    node_size = train_size + len(word_list)
    vocab_length = len(word_list)

    def update_word_and_word_pair_occurrence(q):
        unique_q = list(set(q))
        for i in unique_q:
            try:
                word_occurrence[i] += 1
            except:
                word_occurrence[i] = 1
        for i in range(len(unique_q)):
            for j in range(i+1, len(unique_q)):
                word1 = unique_q[i]
                word2 = unique_q[j]
                word1, word2 = ordered_word_pair(word1, word2)
                try:
                    word_pair_occurrence[(word1, word2)] += 1
                except:
                    word_pair_occurrence[(word1, word2)] = 1
    if not args.easy_copy:
        print("Calculating PMI")
    for ind in range(train_size):
        words = tokenize_sentences[ind]

        q = []
        # push the first (window_size) words into a queue
        for i in range(min(window_size, len(words))):
            q += [word_id_map[words[i]]]
        # update the total number of the sliding windows
        total_W += 1
        # update the number of sliding windows that contain each word and word pair
        update_word_and_word_pair_occurrence(q)

        now_next_word_index = window_size
        # pop the first word out and let the next word in, keep doing this until the end of the document
        while now_next_word_index<len(words):
            q.pop(0)
            q += [word_id_map[words[now_next_word_index]]]
            now_next_word_index += 1
            # update the total number of the sliding windows
            total_W += 1
            # update the number of sliding windows that contain each word and word pair
            update_word_and_word_pair_occurrence(q)

    # calculate PMI for edges
    row = []
    col = []
    weight = []
    for word_pair in word_pair_occurrence:
        i = word_pair[0]
        j = word_pair[1]
        count = word_pair_occurrence[word_pair]
        word_freq_i = word_occurrence[i]
        word_freq_j = word_occurrence[j]
        pmi = log((count * total_W) / (word_freq_i * word_freq_j)) 
        if pmi <=0:
            continue
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)
        row.append(train_size + j)
        col.append(train_size + i)
        weight.append(pmi)
    if not args.easy_copy:
        print("PMI finished.")


    #get each word appears in which document
    word_doc_list = {}
    for word in word_list:
        word_doc_list[word]=[]

    for i in range(train_size):
        doc_words = tokenize_sentences[i]
        unique_words = set(doc_words)
        for word in unique_words:
            exsit_list = word_doc_list[word]
            exsit_list.append(i)
            word_doc_list[word] = exsit_list

    #document frequency
    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    # term frequency
    doc_word_freq = {}

    for doc_id in range(train_size):
        words = tokenize_sentences[doc_id]
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1


    doc_emb = np.zeros((train_size, vocab_length))

    for i in range(train_size):
        words = tokenize_sentences[i]
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            row.append(i)
            col.append(train_size + j)
            idf = log(1.0 * train_size / word_doc_freq[word_list[j]])
            w = freq * idf
            weight.append(w)
            doc_word_set.add(word)
            doc_emb[i][j] = w/len(words)   
            

            
    adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)    
    return adj, doc_emb, word_doc_freq     
