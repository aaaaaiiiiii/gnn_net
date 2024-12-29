import torch.nn as nn
import torch.optim as optim
import optuna
import torch
import random
from torch_geometric.loader import DataLoader
from model import SDRegressionModel
from load_data import load_data
import sys

# --- 設定 ---
num_epochs = 20
num_node_features = 4

# CUDAの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('base_dir: ', file=sys.stderr, end='')
input = input()
train_pkl_file = f'{input}/train.pkl'
test_pkl_file = f'{input}/test.pkl'

# データの読み込み
print('loading train data...', file=sys.stderr)
train_data_set = load_data(train_pkl_file)
print('loading test data...', file=sys.stderr)
test_data_set = load_data(test_pkl_file)

def objective(trial):
    hidden_channels = trial.suggest_int('hidden_channels', 64, 128)
    connection_channels = trial.suggest_int('connection_channels', 64, 128)
    SAGE_num_layers = trial.suggest_int('SAGE_num_layers', 8, 16)
    fc_num_layers = trial.suggest_int('fc_num_layers', 3, 7)
    lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Optunaによるハイパーパラメータチューニング
    model = SDRegressionModel(
        in_channels=num_node_features,
        hidden_channels=hidden_channels,
        SAGE_num_layers=SAGE_num_layers,
        connection_channels=connection_channels,
        fc_num_layers=fc_num_layers
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(num_epochs):
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.sd_index)
                    
            # 損失計算
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.sd_index)
            loss = criterion(output, data.y)
            total_loss += loss.item()

    return total_loss / len(test_loader)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000)

print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
    