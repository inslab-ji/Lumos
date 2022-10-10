import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import argparse
import pickle
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import negative_sampling, train_test_split_edges
from sklearn.metrics import roc_auc_score


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = torch.nn.Linear(out_channels, dataset.num_classes)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = global_mean_pool(x, dict_node, size=data.num_nodes)
        x = F.log_softmax(self.fc(x), dim=1)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False)
        self.fc = torch.nn.Linear(out_channels, dataset.num_classes)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.01, training=self.training).relu()
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.01, training=self.training).relu()
        x = global_mean_pool(x, dict_node, size=data.num_nodes)
        x = F.log_softmax(self.fc(x), dim=1)
        return x


class GCN_unsupervised(torch.nn.Module):
    def __init__(self, hidden_dimension):
        super(GCN_unsupervised, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_dimension)
        self.conv2 = GCNConv(hidden_dimension, hidden_dimension)

    def encode(self, x, train_pos_edge_index):
        x = self.conv1(x, train_pos_edge_index)
        x = F.dropout(x, p=0.01, training=self.training).relu()
        x = self.conv2(x, train_pos_edge_index)
        x = F.dropout(x, p=0.01, training=self.training).relu()
        x = global_mean_pool(x, dict_node, size=data.num_nodes)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index): # only pos and neg edges
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1) # concatenate pos and neg edges
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # dot product
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t() # get adj NxN
        return (prob_adj > 0).nonzero(as_tuple=False).t() # get predicted edge_list


class GAT_unsupervised(torch.nn.Module):
    def __init__(self, hidden_dimension, heads=4):
        super(GAT_unsupervised, self).__init__()
        self.conv1 = GATConv(dataset.num_features, hidden_dimension, heads=heads)
        self.conv2 = GATConv(hidden_dimension*heads, hidden_dimension, heads=1, concat=False)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.01, training=self.training).relu()
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.01, training=self.training).relu()
        x = global_mean_pool(x, dict_node, size=data.num_nodes)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index): # only pos and neg edges
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1) # concatenate pos and neg edges
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # dot product
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t() # get adj NxN
        return (prob_adj > 0).nonzero(as_tuple=False).t() # get predicted edge_list


def train_supervised(model):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test_supervised(model):
    model.eval()
    pred = model(x, edge_index).argmax(dim=-1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


def get_link_labels(pos_edge_index, neg_edge_index):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
    # and the number of zeros is equal to the length of neg_edge_index
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train_unsupervised(model):
    model.train()
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,  # positive edges
        num_nodes=data.num_nodes,  # number of nodes
        num_neg_samples=data.train_pos_edge_index.size(1))  # number of neg_sample equal to number of pos_edges
    optimizer.zero_grad()
    z = model.encode(x, edge_index)  # encode
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)  # decode
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test_unsupervised(model):
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        z = model.encode(x, edge_index)  # encode train
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)  # decode test or val
        link_probs = link_logits.sigmoid()  # apply sigmoid
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)  # get link
        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))  # compute roc_auc score
    return perfs


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Facebook')
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--GNN_model', type=str, default="GCN")
parser.add_argument('--supervised', type=bool, default=False)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.dataset == "Facebook":
    from torch_geometric.datasets import FacebookPagePage
    dataset = FacebookPagePage("./facebook", transform=T.NormalizeFeatures())
    data = dataset[0]
else:
    from torch_geometric.datasets import LastFMAsia
    dataset = LastFMAsia("./lastfm", transform=T.NormalizeFeatures())
    data = dataset[0]
if args.supervised:
    split = RandomNodeSplit(num_val=0.25, num_test=0.25)
    data = split(data)
else:
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    split = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.15)

with open("./"+args.dataset+"/edges.pck", "rb") as file:
    edges = pickle.load(file)
with open("./"+args.dataset+"/tf.pck", "rb") as file:
    tree_features = pickle.load(file)
with open("./"+args.dataset+"/dn.pck", "rb") as file:
    dict_node = pickle.load(file)
edges = [torch.tensor(edge, dtype=torch.int64) for edge in edges]
edge_index = torch.cat(edges, dim=0).t().to(device)
x = torch.tensor(tree_features, dtype=torch.float32).to(device)
dict_node = torch.tensor(dict_node, dtype=torch.int64).to(device)
data = data.to(device)
if args.supervised:
    if args.GNN_model == "GCN":
        model = GCN(dataset.num_features, args.hidden_channels, args.hidden_channels).to(device)
    else:
        model = GAT(dataset.num_features, args.hidden_channels, args.hidden_channels).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_acc = final_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_supervised(model)
        print("Epoch " + str(epoch) + " loss: " + str(loss))
        train_acc, val_acc, tmp_test_acc = test_supervised(model)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        print("Epoch " + str(epoch) + ": Train acc " + str(train_acc))
        print("Val acc " + str(val_acc))
        print("Test acc " + str(tmp_test_acc))
else:
    if args.GNN_model == "GCN":
        model = GCN_unsupervised(args.hidden_channels).to(device)
    else:
        model = GAT_unsupervised(args.hidden_channels).to(device)
    print(model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    best_val_perf = test_perf = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_unsupervised(model)
        val_perf, tmp_test_perf = test_unsupervised(model)
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = tmp_test_perf
        log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_loss, best_val_perf, test_perf))


