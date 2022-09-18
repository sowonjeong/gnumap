import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import sys, os
sys.path.append('../')

from models.baseline_models import MLP, LogReg

def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = adj_rec.cpu()
    # Predict on test set of edges
    preds = []
    for i in range(edges_pos.shape[1]):
        preds.append(sigmoid(adj_rec[edges_pos[0,i], edges_pos[1,i]].item()))

    preds_neg = []
    for i in range(edges_neg.shape[1]):
        preds_neg.append(sigmoid(adj_rec[edges_neg[0,i], edges_neg[1,i]].item()))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score



def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


def plot_acc(train_acc, test_acc=None, xaxis = 'epochs', yaxis = 'accuracy', title = 'Accuracy plot'):
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    if test_acc is not None:
        plt.plot(np.arange(len(train_acc)), train_acc, color='red')
        plt.plot(np.arange(len(test_acc)), test_acc, color='blue')
        plt.legend(['train accuracy', 'test accuracy'], loc='upper right')
    else:
        plt.plot(np.arange(len(train_acc)), train_acc, color='red')
        plt.legend(['train accuracy'], loc='upper right')
    plt.title(title)
    plt.tight_layout()
    plt.show() #show train_acc and test_acc together


def plot_loss(loss, xaxis = 'epochs', yaxis = 'loss', title = 'Loss plot'):
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.plot(np.arange(len(loss)), loss, color='black')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def visualize_graph(G, color, size=300, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         node_color=color, cmap="Set2", node_size = size)
    plt.show()

def visualize_tsne(out, color, size=30, epoch=None, loss = None):
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=size, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss:.4f}', fontsize=16)
    plt.show()


def visualize_umap(out, color, size=30, epoch=None, loss = None):
    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    z = umap_2d.fit_transform(out.detach().cpu().numpy())

    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=size, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss:.4f}', fontsize=16)
    plt.show()

def visualize_pca(out, color, size=30, epoch=None, loss=None):
    h = PCA(n_components=2).fit_transform(out.detach().cpu().numpy())
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(h[:, 0], h[:, 1], s=size, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss:.4f}', fontsize=16)
    plt.show()
