import numpy as np
import torch
from models.baseline_models import MLP, LogReg
from train_utils.evaluation import get_scores


def edge_prediction(embeds, out_dim, train_data, test_data, val_data,
                    lr=0.01, wd=1e-4,
                    patience = 30, max_epochs=3000, print_int=False):
    logreg = LogReg(embeds.shape[1], out_dim)
    opt = torch.optim.Adam(logreg.parameters(), lr=lr, weight_decay=wd)
    output_activation = torch.nn.Sigmoid()
    last_roc = 0
    trigger_times = 0
    best_val_roc = 0
    best_val_ap = 0
    add_neg_samples = True
    loss_fn = torch.nn.BCELoss()
    results = []
    best_epoch  = 0
    pos_edge_index = train_data.pos_edge_label_index
    neg_edge_index = train_data.neg_edge_label_index
    for epoch in range(max_epochs):
        logreg.train()
        opt.zero_grad()

        #### 1st alternative:
        logits_temp = logreg(embeds)
        logits = output_activation(torch.mm(logits_temp, logits_temp.t()))
        loss = (loss_fn(logits[pos_edge_index[0,:],pos_edge_index[1,:]], torch.ones(pos_edge_index.shape[1]))+
                    loss_fn(logits[neg_edge_index[0,:],neg_edge_index[1,:]], torch.zeros(neg_edge_index.shape[1])))
        loss.backward(retain_graph=True)
        opt.step()

        logreg.eval()
        with torch.no_grad():
            try:
                val_roc, val_ap = get_scores(val_data.pos_edge_label_index, val_data.neg_edge_label_index, logits)
            except:
                val_roc, val_ap  = np.nan, np.nan
            try:
                test_roc, test_ap = get_scores(test_data.pos_edge_label_index, test_data.neg_edge_label_index, logits)
            except:
                test_roc, test_ap = np.nan, np.nan
            try:
                train_roc, train_ap = get_scores(train_data.pos_edge_label_index, train_data.neg_edge_label_index, logits)
            except:
                train_roc, train_ap = np.nan, np.nan

            if np.isnan(val_roc):
                break
            if (np.isnan(val_roc) ==False) & (val_roc >= best_val_roc):
                best_val_roc = val_roc

            current_roc = val_roc
            results += [[epoch, val_ap, val_roc, test_ap, test_roc, train_ap, train_roc ]]
            if print_int:
                print([epoch, val_ap, val_roc, test_ap, test_roc, train_ap, train_roc ])
            if current_roc <= last_roc:
                trigger_times += 1
                #print('Trigger Times:', trigger_times)
                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    break
            else:
                #print('trigger times: 0')
                trigger_times = 0
                last_roc= current_roc
                best_epoch = epoch
    return(logreg, results, best_epoch)
