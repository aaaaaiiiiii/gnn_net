import torch
from torch_geometric.nn.models import GraphSAGE
    
class GNNLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout):
        super(GNNLayer, self).__init__()
        self.layer = GraphSAGE(in_channels, hidden_channels, num_layers, out_channels, dropout)
    
    def forward(self, x, edge_index):
        x = self.layer(x, edge_index)
        return x
        
class SDFCLayer(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(SDFCLayer, self).__init__()
        self.fc1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, 1)
        
    def forward(self, z, edge_index):
        link_emb = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        x = self.fc1(link_emb).relu()
        x = self.fc2(x)

        return x.view(-1)

class SDRegressionModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, SAGE_num_layers, connection_channels, dropout):
        super(SDRegressionModel, self).__init__()
        self.gnn = GNNLayer(in_channels, hidden_channels, SAGE_num_layers, connection_channels, dropout)
        self.sd_fc = SDFCLayer(connection_channels)

    def forward(self, x, edge_index, sd_index):
        z = self.gnn(x, edge_index)
        out = self.sd_fc(z, sd_index)
        return out