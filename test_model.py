
import torch
from load_data import load_data
from model import SDRegressionModel


input = input('base_dir: ')

pkl_file = f'{input}/test.pkl'
model_file = f'{input}/model.pth'

# データの読み込み
test_data_set = load_data(pkl_file)

test_data_set = test_data_set[1000]

# --- モデルの定義 ---
hidden_channels = 108
SAGE_num_layers = 5
connection_channels = 90
dropout = 0.15
lr = 0.00034
batch_size = 64
fc_num_layers = 3

model = SDRegressionModel(in_channels=4, hidden_channels=hidden_channels, SAGE_num_layers=SAGE_num_layers, connection_channels=connection_channels, fc_num_layers=fc_num_layers)
model.load_state_dict(torch.load(model_file))

# --- テスト ---

model.eval()
with torch.no_grad():
    data = test_data_set
    out = model(data.x, data.edge_index, data.sd_index)
    for i in range(len(out)):
        print(f'{data.y[i]} {out[i]} {round(out[i].item()) == data.y[i]}')
