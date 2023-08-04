import numpy as np
import torch
from models.baseline_models import MLP, LogReg


def node_prediction(embeds, out_dim, y, train_mask, test_mask, val_mask,
                    lr=0.01, wd=1e-4,  n_layers=1,
                    patience = 30, max_epochs=3000):

    if n_layers > 1:
        node_classifier = MLP(embeds.shape[1], int(np.max([int(embeds.shape[1])/2, out_dim])),
                              out_dim,  n_layers=2)
    else:
        node_classifier = LogReg(embeds.shape[1], out_dim)
    train_labels = y[train_mask]
    test_labels = y[test_mask]
    val_labels = y[val_mask]
    optimizer_temp = torch.optim.Adam(node_classifier.parameters(), lr=lr,
                                      weight_decay=wd)
    res_temp = []
    trigger_times = 0
    last_acc = 0
    best_epoch = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch_temp in range(max_epochs):
        node_classifier.train();
        optimizer_temp.zero_grad();
        out = node_classifier(embeds);
        loss_temp = loss_fn (out[train_mask], train_labels);
        loss_temp.backward()
        optimizer_temp.step()

        preds = torch.argmax(out, dim=1)
        acc_train = torch.sum(preds[train_mask] == train_labels).float() / train_labels.shape[0]
        acc = torch.sum(preds[test_mask] == test_labels).float() / test_labels.shape[0]
        val_acc = torch.sum(preds[val_mask] == val_labels).float() / val_labels.shape[0]
        res_temp += [[epoch_temp, loss_temp.cpu().item(), acc_train.item(), val_acc.item() ,  acc.item()]]


        current_acc = val_acc
        if current_acc <= last_acc:
            trigger_times += 1
            #print('Trigger Times:', trigger_times)
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                break
        else:
            trigger_times = 0
            last_acc = current_acc
            best_epoch = epoch_temp

    return(node_classifier, res_temp, best_epoch)
