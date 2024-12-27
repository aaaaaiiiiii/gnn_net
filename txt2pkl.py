import numpy as np
import os
import torch
from torch_geometric.data import Data
from tqdm import tqdm

def adj_matrix_to_edge_index(adj_matrix):
    edge_index = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i][j] == 1:
                edge_index.append([i, j])
    return np.array(edge_index).T

def node_degree(adj_matrix):
    return np.sum(adj_matrix, axis=1)

def edge_index_to_node_degree(edge_index):
    node_degree = np.zeros((edge_index.max() + 1))
    for i in range(edge_index.shape[1]):
        node_degree[edge_index[0][i]] += 1
        node_degree[edge_index[1][i]] += 1
    return node_degree

def file_to_data(file_path, edge_index):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    node_feature1 = np.array(lines[0].strip().split(','), dtype=int).reshape(-1, 1)
    node_feature2 = np.array(lines[1].strip().split(','), dtype=int).reshape(-1, 1)
    node_feature3 = np.array(lines[2].strip().split(','), dtype=int).reshape(-1, 1)
    node_feature4 = np.array(lines[3].strip().split(','), dtype=int).reshape(-1, 1)


    node_feature = np.concatenate((node_feature1, node_feature2, node_feature3, node_feature4), axis=1)
    
    sd_index_src = np.array(lines[4].strip().split(','), dtype=int)
    sd_index_dst = np.array(lines[5].strip().split(','), dtype=int)
    sd_index = np.stack((sd_index_src, sd_index_dst), axis=0)
    
    y = np.array(lines[6].strip().split(','), dtype=int)

    node_feature = torch.tensor(node_feature, dtype=torch.float32)
    edge_index = edge_index.clone().detach()
    sd_index = torch.tensor(sd_index, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float32)

    # # サンプリングするインデックスをランダムに取得
    # mask = random.sample(range(sd_index.size(1)), sd_batch_size)  # data.sd_indexは(2, 600*batch_size)なので、1次元目でサンプリング
    # mask = torch.tensor(mask, dtype=torch.long)

    # # サンプリングされたSDペアとラベルを取得
    # sampled_sd_index = sd_index[:, mask]  # (2, sd_batch_size)
    # sampled_y = y[mask]  # (sd_batch_size)

    return Data(x=node_feature, edge_index=edge_index, sd_index=sd_index, y=y)



def topology_file_to_edge_index(file_path):
    '''
    ファイルからエッジインデックスを取得する

    Parameters
    ----------
    file_path : str
        隣接行列形式のテキストファイルへのパス

    Returns
    -------
    edge_index : torch.tensor
        エッジインデックス
    
    '''

    with open(file_path, 'r') as f:
        lines = f.readlines()

    adj_matrix = np.array([list(map(int, line.strip().split(','))) for line in lines])

    edge_index = adj_matrix_to_edge_index(adj_matrix)

    return torch.tensor(edge_index, dtype=torch.long)        
    

def create_pkl(dir, save_path):
    train_data_set = []
    test_data_set = []

    train_dir = f'{dir}/train'
    test_dir = f'{dir}/test'

    adj_matrix_file_path = f'{dir}/topology.txt'

    edge_index = topology_file_to_edge_index(adj_matrix_file_path)

    for file in tqdm(os.listdir(train_dir), desc="Processing train files"):
        file_path = f'{train_dir}/{file}'
        geo_data = file_to_data(file_path, edge_index)
        train_data_set.append(geo_data)

    for file in tqdm(os.listdir(test_dir), desc="Processing test files"):
        file_path = f'{test_dir}/{file}'
        geo_data = file_to_data(file_path, edge_index)
        test_data_set.append(geo_data)

    with open(f'{dir}/train.pkl', 'wb') as f:
        torch.save(train_data_set, f)
        print(f'saved: {dir}/train.pkl')

    with open(f'{dir}/test.pkl', 'wb') as f:
        torch.save(test_data_set, f)
        print(f'saved: {dir}/test.pkl')

    print(f'num of train data: {len(train_data_set)}')
    print(f'num of test data: {len(test_data_set)}')


if __name__ == '__main__':
    base_dir = input('base_dir: ')
    save_path = f'{base_dir}/{base_dir}.pkl'
    create_pkl(base_dir, save_path)