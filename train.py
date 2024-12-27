import torch
import torch.nn as nn
import torch.optim as optim
import sys
from torch_geometric.loader import DataLoader

from model import SDRegressionModel
from load_data import load_data

# CUDAの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# --- 設定 ---
num_epochs = 200
num_node_features = 4

# --- ハイパーパラメータ ---
hidden_channels = 108
SAGE_num_layers = 5
connection_channels = 90
lr = 0.00034
batch_size = 64
fc_num_layers = 3

# --- モデルの定義 ---
model = SDRegressionModel(
    in_channels=num_node_features,
    hidden_channels=hidden_channels,
    SAGE_num_layers=SAGE_num_layers,
    connection_channels=connection_channels,
    fc_num_layers=fc_num_layers
).to(device)

# --- 学習設定 ---
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()


print('base_dir: ', file=sys.stderr, end='')
input = input()
train_pkl_file = f'{input}/train.pkl'
test_pkl_file = f'{input}/test.pkl'

# データの読み込み
print('loading train data...', file=sys.stderr)
train_data_set = load_data(train_pkl_file)
print('loading test data...', file=sys.stderr)
test_data_set = load_data(test_pkl_file)

# PyTorch Geometric用 DataLoaderの作成
train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)


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