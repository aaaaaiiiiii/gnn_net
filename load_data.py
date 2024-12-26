import torch

def load_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = torch.load(f, weights_only=False)
        return data

if __name__ == '__main__':
    dir = input('base_dir: ')
    pkl_file = f'{dir}/{dir}.pkl'
    train_data, test_data = load_data(pkl_file)
    print(f'train_data: {len(train_data)}')
    print(f'test_data: {len(test_data)}')    