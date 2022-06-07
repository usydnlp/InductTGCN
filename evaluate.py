import numpy as np
from math import log
import torch
import torch.nn.functional as F

def get_weights_hidden(model,features,adj,train_size):
    model.eval()
    hidden_output, _ = model(features, adj)
    weights1 = model.gc1.weight
    bias1 = model.gc1.bias
    weights1.require_grad = False
    bias1.require_grad = False
    weights2 = model.gc2.weight
    bias2 = model.gc2.bias
    weights2.require_grad = False
    bias2.require_grad = False
    return [hidden_output[train_size:], weights1, bias1, weights2, bias2]

def get_test_emb(tokenize_test_sentences, word_id_map, vocab_length, word_doc_freq, word_list, train_size):
    test_size = len(tokenize_test_sentences)
    test_emb = [[0]*vocab_length for _ in range(test_size)]
    tokenized_test_edge = [[0]*vocab_length+[1] for _ in range(test_size)]
    for i in range(test_size):
        tokenized_test_sample = tokenize_test_sentences[i]
        word_freq_list = [0]*vocab_length
        for word in tokenized_test_sample:
            if word in word_id_map:
                word_freq_list[word_id_map[word]]+=1
            
        for word in tokenized_test_sample:
            if word in word_id_map:
                j = word_id_map[word]
                freq = word_freq_list[j]   
                idf = log(1.0 * train_size / word_doc_freq[word_list[j]])
                w = freq * idf
                test_emb[i][j] = w/len(tokenized_test_sample)
                tokenized_test_edge[i][j] = w
         
        # norm_item_temp = (1/norm_item[train_size:]).tolist()+[np.sqrt(np.sum(tokenized_test_edge[i]))]
        # tokenized_test_edge[i] = tokenized_test_edge[i]/np.array(norm_item_temp)

    tokenized_test_edge = np.array(tokenized_test_edge)
    return test_emb, tokenized_test_edge


def test_model(model, test_emb, tokenized_test_edge,model_weights_list,device):
    hidden_output, weights1, bias1, weights2, bias2 = model_weights_list
    test_result = []
    for ind in range(len(test_emb)):
        tokenized_test_edge_temp = torch.FloatTensor([tokenized_test_edge[ind]]).to(device)
        hidden_temp = F.relu(torch.mm(tokenized_test_edge_temp, torch.vstack((weights1, torch.mm(torch.FloatTensor([test_emb[ind]]).to(device), weights1)))) + bias1)
        test_hidden_temp = torch.cat((hidden_output,hidden_temp))
        test_output_temp = torch.mm(tokenized_test_edge_temp, torch.mm(test_hidden_temp, weights2)) + bias2
        predict_temp = torch.argmax(test_output_temp).cpu().detach().tolist()
        test_result.append(predict_temp)
    return test_result

