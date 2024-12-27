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
    def __init__(self, hidden_channels, num_layers):
        super(SDFCLayer, self).__init__()

        layers = []
        
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(hidden_channels * 2, hidden_channels * 2))
            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(hidden_channels * 2, 1))

        self.fc_layers = torch.nn.Sequential(*layers)
        
    def forward(self, z, edge_index):
        link_emb = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        x = self.fc_layers(link_emb)
        return x.view(-1)

class SDRegressionModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, SAGE_num_layers, connection_channels, fc_num_layers, dropout=0):
        super(SDRegressionModel, self).__init__()
        self.gnn = GNNLayer(in_channels, hidden_channels, SAGE_num_layers, connection_channels, dropout)
        self.sd_fc = SDFCLayer(connection_channels, fc_num_layers)

    def forward(self, x, edge_index, sd_index):
        z = self.gnn(x, edge_index)
        out = self.sd_fc(z, sd_index)
        return out