import time
import numpy as np
from utils import cal_accuracy

def train_model(args, model, optimizer, criterion, features, adj, labels, idx_train, idx_val, show_result = True):
    val_loss = []
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        _, output= model(features, adj)
        loss_train = criterion(output[idx_train], labels[idx_train])
        acc_train = cal_accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        _, output = model(features, adj)

        loss_val = criterion(output[idx_val], labels[idx_val])
        val_loss.append(loss_val.item())
        acc_val = cal_accuracy(output[idx_val], labels[idx_val])
        if show_result:
            print(  'Epoch: {:04d}'.format(epoch+1),
                    'loss_train: {:.4f}'.format(loss_train.item()),
                    'acc_train: {:.4f}'.format(acc_train),
                    'loss_val: {:.4f}'.format(loss_val.item()),
                    'acc_val: {:.4f}'.format(acc_val),
                    'time: {:.4f}s'.format(time.time() - t))
        
        if epoch > args.early_stopping and np.min(val_loss[-args.early_stopping:]) > np.min(val_loss[:-args.early_stopping]) :
            if show_result:
                print("Early Stopping...")
            break

