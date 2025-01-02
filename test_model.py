
import torch
from load_data import load_data
from model import SDRegressionModel
from torch_geometric.loader import DataLoader

# CUDAの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_data_dir = input('test_data_dir: ')
model_dir = input('model_dir: ')

pkl_file = f'{test_data_dir}/test.pkl'
model_file = f'{model_dir}/model.pth'

# データの読み込み
test_data_set = load_data(pkl_file)


# --- ハイパーパラメータ ---
hidden_channels = 121
connection_channels = 110
SAGE_num_layers = 9
fc_num_layers = 5
lr = 0.00041098201353148003
batch_size = 16

model = SDRegressionModel(in_channels=4, hidden_channels=hidden_channels, SAGE_num_layers=SAGE_num_layers, connection_channels=connection_channels, fc_num_layers=fc_num_layers)
model.load_state_dict(torch.load(model_file))

test_data_loader = DataLoader(test_data_set, batch_size=1024, shuffle=False)

# --- テスト ---
model.eval()
with torch.no_grad():
    total_loss = 0
    for data in test_data_loader:
        out = model(data.x, data.edge_index, data.sd_index)
        loss = torch.nn.MSELoss()(out, data.y)
        total_loss += loss.item()
    print(f'Loss: {total_loss / len(test_data_loader)}')


# --- テスト ---

# model.eval()
# with torch.no_grad():
#     while True:
#         print('index: ')
#         index = int(input())
#         data = test_data_set[index]
#         out = model(data.x, data.edge_index, data.sd_index)
#         for i in range(len(out)):
#             print(f'{data.y[i]} {out[i]} {round(out[i].item()) == data.y[i]}')
        
#         accuracy = sum([round(out[i].item()) == data.y[i] for i in range(len(out))]) / len(out)
#         print(f'accuracy: {accuracy}')
