import numpy as np
import os
import pickle
import torch
from torch_geometric.data import Data
import random

def file_to_data(file_path, sd_batch_size):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    node_feature1 = np.array(lines[0].strip().split(','), dtype=int).reshape(-1, 1)
    node_feature2 = np.array(lines[1].strip().split(','), dtype=int).reshape(-1, 1)
    node_feature = np.concatenate((node_feature1, node_feature2), axis=1)

    edge_index_src = np.array(lines[2].strip().split(','), dtype=int)
    edge_index_dst = np.array(lines[3].strip().split(','), dtype=int)
    edge_index = np.stack((edge_index_src, edge_index_dst), axis=0)

    sd_index_src = np.array(lines[4].strip().split(','), dtype=int)
    sd_index_dst = np.array(lines[5].strip().split(','), dtype=int)
    sd_index = np.stack((sd_index_src, sd_index_dst), axis=0)
    
    
    y = np.array(lines[6].strip().split(','), dtype=int)

    node_feature = torch.tensor(node_feature, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    sd_index = torch.tensor(sd_index, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float32)

    # # サンプリングするインデックスをランダムに取得
    # mask = random.sample(range(sd_index.size(1)), sd_batch_size)  # data.sd_indexは(2, 600*batch_size)なので、1次元目でサンプリング
    # mask = torch.tensor(mask, dtype=torch.long)

    # # サンプリングされたSDペアとラベルを取得
    # sampled_sd_index = sd_index[:, mask]  # (2, sd_batch_size)
    # sampled_y = y[mask]  # (sd_batch_size)

    return Data(x=node_feature, edge_index=edge_index, sd_index=sd_index, y=y)

def create_pkl(dir, save_path):
    train_data_set = []
    test_data_set = []

    train_dir = f'{dir}/train'
    test_dir = f'{dir}/test'

    for file in os.listdir(train_dir):
        file_path = f'{train_dir}/{file}'
        geo_data = file_to_data(file_path, 32)
        train_data_set.append(geo_data)

    for file in os.listdir(test_dir):
        file_path = f'{test_dir}/{file}'
        geo_data = file_to_data(file_path, 600)
        test_data_set.append(geo_data)

    data_set = {'train': train_data_set, 'test': test_data_set}

    with open(save_path, 'wb') as f:
        torch.save(data_set, f)
        print(f'saved: {save_path}')

    for key, value in data_set.items():
        print(f'{key}: {len(value)}')

if __name__ == '__main__':
    base_dir = input('base_dir: ')
    save_path = f'{base_dir}/{base_dir}.pkl'
    create_pkl(base_dir, save_path)

    