import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(EdgeDecoder, self).__init__()
        self.fc1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z, edge_index):
        edge_emb = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        edge_emb = self.fc1(edge_emb).relu()
        edge_emb = self.fc2(edge_emb)

        return edge_emb.view(-1)
    
class SDRegressionModel(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(SDRegressionModel, self).__init__()
        self.encoder = GraphEncoder(hidden_channels, hidden_channels)
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x, edge_index, sd_index):
        z = self.encoder(x, edge_index)
        out = self.decoder(z, sd_index)
        return out

