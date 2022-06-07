import scipy.sparse as sp
import numpy as np
import torch
from sklearn.metrics import accuracy_score

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo(), d_inv_sqrt
    
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def decide_device(args):
    if args.use_gpu:
        if torch.cuda.is_available():
            if not args.easy_copy:
                print("Use CUDA")
            device = torch.device("cuda")
        else:
            if not args.easy_copy:
                print("CUDA not avaliable, use CPU instead")
            device = torch.device("cpu")
    else:
        if not args.easy_copy:
            print("Use CPU")
        device = torch.device("cpu")
    return device


def generate_train_val(args, train_size, train_pro=0.9):
    real_train_size = int(train_pro*train_size)
    val_size = train_size-real_train_size

    if args.shuffle_seed!=None:
        np.random.seed(args.shuffle_seed)
    idx_train = np.random.choice(train_size, real_train_size,replace=False)
    idx_train.sort()
    idx_val = []
    pointer = 0
    for v in range(train_size):
        if pointer<len(idx_train) and idx_train[pointer] == v:
            pointer +=1
        else:
            idx_val.append(v)
    return idx_train, idx_val    


def generate_train_val(args, train_size, train_pro=0.9):
    real_train_size = int(train_pro*train_size)
    val_size = train_size-real_train_size

    if args.shuffle_seed!=None:
        np.random.seed(args.shuffle_seed)
    idx_train = np.random.choice(train_size, real_train_size,replace=False)
    idx_train.sort()
    idx_val = []
    pointer = 0
    for v in range(train_size):
        if pointer<len(idx_train) and idx_train[pointer] == v:
            pointer +=1
        else:
            idx_val.append(v)
    return idx_train, idx_val       


def cal_accuracy(predictions,labels):
    pred = torch.argmax(predictions,-1).cpu().tolist()
    lab = labels.cpu().tolist()
    return accuracy_score(lab,pred)