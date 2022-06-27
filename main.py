from build_dataset import get_dataset
from preprocess import encode_labels,preprocess_data
from build_graph import get_adj
from train import train_model
from utils import *
from model import GCN
from evaluate import get_weights_hidden, get_test_emb, test_model
import argparse
import torch
import torch.optim as optim
import scipy.sparse as sp
import torch.nn as nn
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='R8', help='Dataset string: R8, R52, OH, 20NGnew, MR')
parser.add_argument('--train_size',  type=float, default=1, help='If it is larger than 1, it means the number of training samples. If it is from 0 to 1, it means the proportion of the original training set.')
parser.add_argument('--test_size',  type=float, default=1, help='If it is larger than 1, it means the number of training samples. If it is from 0 to 1, it means the proportion of the original training set.')
parser.add_argument('--remove_limit', type=int, default=2, help='Remove the words showing fewer than 2 times')
parser.add_argument('--use_gpu', type=int, default=1, help='Whether to use GPU, 1 means True and 0 means False. If True and no GPU available, will use CPU instead.')
parser.add_argument('--shuffle_seed',type = int, default = None, help="If not specified, train/val is shuffled differently in each experiment")
parser.add_argument('--hidden_dim',type = int, default = 200, help="The hidden dimension of GCN model")
parser.add_argument('--dropout',type = float, default = 0.5, help="The dropout rate of GCN model")
parser.add_argument('--learning_rate',type = float, default = 0.02, help="Learning rate")
parser.add_argument('--weight_decay',type = float, default = 0, help="Weight decay, normally it is 0")
parser.add_argument('--early_stopping',type = int, default = 10, help="Number of epochs of early stopping.")
parser.add_argument('--epochs',type = int, default = 200, help="Number of maximum epochs")
parser.add_argument('--multiple_times',type = int, default = 10, help="Running multiple experiments, each time the train/val split is different")
parser.add_argument('--easy_copy',type = int, default = 1, help="For easy copy of the experiment results. 1 means True and 0 means False.")

args = parser.parse_args()

device = decide_device(args)

# Get dataset
sentences, labels, train_size, test_size = get_dataset(args)
train_sentences = sentences[:train_size]
test_sentences = sentences[train_size:]
train_labels = labels[:train_size]
test_labels = labels[train_size:]

# Preprocess text and labels
labels, num_class = encode_labels(train_labels, test_labels, args)
labels = torch.LongTensor(labels).to(device)
tokenize_sentences, word_list = preprocess_data(train_sentences, test_sentences, args)
vocab_length = len(word_list)
word_id_map = {}
for i in range(vocab_length):
    word_id_map[word_list[i]] = i
if not args.easy_copy:
    print("There are", vocab_length, "unique words in total.")   

# Generate Graph
adj, doc_emb, word_doc_freq = get_adj(tokenize_sentences,train_size,word_id_map,word_list,args)
adj, norm_item = normalize_adj(adj + sp.eye(adj.shape[0]))
adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
features = torch.FloatTensor(doc_emb).to(device)

criterion = nn.CrossEntropyLoss()

# Generate Test input
test_emb, tokenized_test_edge = get_test_emb(tokenize_sentences[train_size:], word_id_map, vocab_length, word_doc_freq, word_list, train_size)

if not args.easy_copy:
    # Generate train/val dataset
    idx_train, idx_val = generate_train_val(args, train_size)

    # Genrate Model
    model = GCN(nfeat=vocab_length, nhid=args.hidden_dim, nclass=num_class, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Train the model
    train_model(args, model, optimizer, criterion, features, adj, labels, idx_train, idx_val)

    # Test
    if not args.easy_copy:
        print("Predicting on test set.") 
    model_weights_list = get_weights_hidden(model,features,adj,train_size)
    test_result = test_model(model, test_emb, tokenized_test_edge,model_weights_list,device)
    print(classification_report(labels[train_size:].cpu(),test_result,digits = 4))

if args.multiple_times:
    test_acc_list = []
    for t in range(args.multiple_times):
        if not args.easy_copy:
            print("Round",t+1)
        model = GCN(nfeat=vocab_length, nhid=args.hidden_dim, nclass=num_class, dropout=args.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        idx_train, idx_val = generate_train_val(args, train_size)
        train_model(args, model, optimizer, criterion, features, adj, labels, idx_train, idx_val, show_result=False)
        model_weights_list = get_weights_hidden(model,features,adj,train_size)
        test_result = test_model(model, test_emb, tokenized_test_edge,model_weights_list,device)
        test_acc_list.append(accuracy_score(labels[train_size:].cpu(),test_result))
    if args.easy_copy:

        print("%.4f"%np.mean(test_acc_list), end = ' ± ')
        print("%.4f"%np.std(test_acc_list))

    else: 
        for t in test_acc_list:
            print("%.4f"%t)       
        print("Test Accuracy:",np.round(test_acc_list,4).tolist())
        print("Mean:%.4f"%np.mean(test_acc_list))
        print("Std:%.4f"%np.std(test_acc_list))
