import torch
import torch.nn as nn
import torch.optim as optim
import sys
import random
from torch_geometric.loader import DataLoader

from model import SDRegressionModel
from load_data import load_data


num_epochs = 200

# --- ハイパーパラメータ ---
hidden_channels = 121
# num_heads = 6
lr = 0.001882383
batch_size = 128
sd_batch_size = 32

# CUDAの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

input = input('base_dir: ')
pkl_file = f'{input}/{input}.pkl'

# データの読み込み

train_data_set, test_data_set = load_data(pkl_file)

# PyTorch Geometric用 DataLoaderの作成

train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)

# --- モデルの定義 ---
model = SDRegressionModel(hidden_channels=hidden_channels).to(device)

# --- 学習設定 ---
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# --- 学習 ---
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for data in train_loader:
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.sd_index)
                
        # 損失計算
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch: {epoch}, Loss: {total_loss / len(train_loader)}')
    sys.stdout.flush()

# --- テスト ---
model.eval()
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.sd_index)
        loss = criterion(out, data.y)
    print(f'Test Loss: {loss.item()}')

# --- モデルの保存 ---
torch.save(model.state_dict(), f'{input}/model.pth')