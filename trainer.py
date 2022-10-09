import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import argparse
import mcmc
import tree_constructor
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import sort_edge_index


def neighbours(edges, num_nodes):
    src, tag = edges
    n = []
    si = 0
    for i in range(num_nodes):
        l = []
        while si < len(src) and src[si] == i:
            l.append(int(tag[si]))
            si += 1
        n.append(l)
    return n


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.01, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.01, training=self.training)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False)
        self.fc = torch.nn.Linear(hidden_channels, data.num_classes)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.01, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.01, training=self.training)
        x = global_mean_pool(x, data.num_nodes)
        x = F.log_softmax(self.fc(x), dim=1)
        return x


def train_supervised(model):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test_supervised():
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Facebook')
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--mcmcepochs', type=int, default=100000)
parser.add_argument('--GNN_model', type=str, default="GCN")
parser.add_argument('--supervised', type=bool, default=True)
parser.add_argument('--epsilon', type=float, default=2)
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
edge_index = sort_edge_index(data.edge_index)
split = RandomNodeSplit(num_val=0.25, num_test=0.25)
data = split(data)
neighs = neighbours(edge_index, data.num_nodes)
init_s = mcmc.init(neighs)
final_s = mcmc.mcmc(args.mcmcepochs, init_s)
edges, tree_features, dict_node = tree_constructor.construct_tree(final_s, data.x)
edge_index = torch.tensor(edges).t().to(device)
x = torch.tensor(tree_features).to(device)
data = data.to(device)
if args.GNN_model == "GCN":
    model = GCN(dataset.num_features, args.hidden_channels, args.hidden_channels).to(device)
else:
    model = GAT(dataset.num_features, args.hidden_channels, args.hidden_channels).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
best_val_acc = final_test_acc = 0
for epoch in range(1, args.epochs + 1):
    loss = train_supervised(model)
    train_acc, val_acc, tmp_test_acc = test_supervised()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print("Epoch "+str(epoch)+": Train acc "+str(train_acc))
    print("Val acc"+str(val_acc))
    print("Test acc "+str(tmp_test_acc))
