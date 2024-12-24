
import torch
from load_data import load_data
from model import SDRegressionModel


input = input('base_dir: ')

pkl_file = f'{input}/{input}.pkl'
model_file = f'{input}/model.pth'

# データの読み込み
train_data_set, test_data_set = load_data(pkl_file)

test_data_set = test_data_set[0]

# --- モデルの定義 ---
model = SDRegressionModel(hidden_channels=121)
model.load_state_dict(torch.load(model_file))

# --- テスト ---

model.eval()
with torch.no_grad():
    data = test_data_set
    out = model(data.x, data.edge_index, data.sd_index)
    for i in range(len(out)):
        print(f'{data.y[i]} {out[i]}')
