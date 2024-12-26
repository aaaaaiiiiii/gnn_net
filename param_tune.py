import torch.nn as nn
import torch.optim as optim
import optuna
import torch
import random
from torch_geometric.loader import DataLoader
from model import SDRegressionModel
from load_data import load_data
import sys

num_epochs = 50

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
    hidden_channels = trial.suggest_int('hidden_channels', 16, 128)
    # num_heads = trial.suggest_int('num_heads', 1, 8)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024])
    # sd_batch_size = trial.suggest_categorical('sd_batch_size', [64, 128, 256, 512])
    
    # Optunaによるハイパーパラメータチューニング
    model = SDRegressionModel(hidden_channels=hidden_channels).to(device)
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
study.optimize(objective, n_trials=100)

print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
    