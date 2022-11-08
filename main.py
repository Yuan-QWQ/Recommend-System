import dataset
import torch
import pandas as pd
from torch.utils.data import DataLoader
from MyModel.LR import LogisticRegressionModel
from MyModel.FM import FactorizationMachineModel
from MyModel.FFM import FieldAwareFactorizationMachine
from MyModel.WideDeep import WideDeepModel
from MyModel.DeepFM import DeepFactorizationMachineModel
from MyModel.FNN import FactorizationSupportedNeuralNetworkModel
from MyModel.PNN import ProductBasedNeuralNetworkModel
from MyModel.NFM import NeuralFactorizationMachinesModel
from MyModel.DCN import DeepCrossNetworkModel
from MyModel.AFM import AttentionalFactorizationMachinesModel
from MyModel.xDeepFM import xDeepFM
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            # torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def test(model, data_loader):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in data_loader:
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


def train(model, optimizer, train_data_loader, criterion,  log_interval):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 1):
        inputs, labels = data

        optimizer.zero_grad()
        # 求loss
        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        # 梯度反向传播
        loss.backward()
        # 由梯度，更新参数
        optimizer.step()
        # 可视化
        # print statistics
        running_loss += loss.item()
        if i % log_interval == 0:  # print every 2000 mini-batches
            print('batch: %5d loss: %.3f' %
                  (i, running_loss / log_interval))
            running_loss = 0.0


if __name__ == '__main__':
    file_path = 'data/train.txt'
    dataset = dataset.CriteoDataset(file_path, 2000)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.2)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=200)
    valid_data_loader = DataLoader(valid_dataset, batch_size=200)
    test_data_loader = DataLoader(test_dataset, batch_size=200)

    field_dims = dataset.field_dims
    model = xDeepFM(field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.05, weight_decay=0.001)
    save_dir = ''
    model_name = ''
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}.pt')
    for epoch in range(100):
        train(model, optimizer, train_data_loader, criterion, log_interval=8)
        auc = test(model, valid_data_loader)
        print('epoch:', epoch+1, 'validation: auc:', auc)
        print('\n')
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break

    auc = test(model, valid_data_loader)
    print(f'test auc: {auc}')


