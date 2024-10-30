##用新预测的特征去训练多模态模型
import copy
import glob
import os
import random

import torch
from scipy.io import savemat
from sklearn import preprocessing
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import IterableDataset, DataLoader
import scipy.io as sio
from tqdm import tqdm

from model import TransformerEncoder

class MyIterableDataset(IterableDataset):
    def __init__(self, file_list,dataset_name):
        super(MyIterableDataset, self).__init__()
        self.file_list = file_list
        self.dataset_name=dataset_name


    def parse_file(self, file):
        file_list = "/home/yjliang/PycharmProjects/CLIP/generate_data/final_beat_clip_fea/mean_fea/"+self.dataset_name+"/"+file
        data = sio.loadmat(file_list)
        images = data['final_fea']



        ##还是先要判断是验证集还是训练集
        keyword1 = "train"
        keyword2 = "valid"
        keyword3 = "test"

        if keyword1 in file_list:
            current_name = file
            #label_file_name = "/home/yjliang/PycharmProjects/CLIP/need_data/_5407_divide_dataset_label/train/" + current_name
            label_file_name = "/home/yjliang/PycharmProjects/CLIP/need_data/_5407_divide_dataset_function_label/train/" + current_name
            #seq_file_name = "/home/yjliang/PycharmProjects/structure_bert/generate_data/seq_struc_fea/position/train_norm/" + current_name
            #go_file_name = "/home/yjliang/PycharmProjects/GET_GO_FEA/generate_data/go_my_transformer_fea_norm/position/train/" + current_name
            seq_file_name = "/home/yjliang/PycharmProjects/structure_bert/generate_data/seq_struc_fea/function/train_norm/" + current_name
            go_file_name ="/home/yjliang/PycharmProjects/GET_GO_FEA/generate_data/go_my_transformer_fea_norm/function/train/"+ current_name

        elif keyword2 in file_list:
            current_name = file
            #label_file_name = "/home/yjliang/PycharmProjects/CLIP/need_data/_5407_divide_dataset_label/valid/" + current_name
            label_file_name = "/home/yjliang/PycharmProjects/CLIP/need_data/_5407_divide_dataset_function_label/valid/" + current_name
            #seq_file_name = "/home/yjliang/PycharmProjects/structure_bert/generate_data/seq_struc_fea/position/valid_norm/" + current_name
            seq_file_name = "/home/yjliang/PycharmProjects/structure_bert/generate_data/seq_struc_fea/function/valid_norm/" + current_name
            #go_file_name = "/home/yjliang/PycharmProjects/GET_GO_FEA/generate_data/go_my_transformer_fea_norm/position/valid/" + current_name
            go_file_name = "/home/yjliang/PycharmProjects/GET_GO_FEA/generate_data/go_my_transformer_fea_norm/function/valid/" + current_name

        elif keyword3 in file_list:
            current_name = file
            #label_file_name = "/home/yjliang/PycharmProjects/CLIP/need_data/_5407_divide_dataset_label/test/" + current_name
            label_file_name = "/home/yjliang/PycharmProjects/CLIP/need_data/_5407_divide_dataset_function_label/test/" + current_name
            #seq_file_name = "/home/yjliang/PycharmProjects/structure_bert/generate_data/seq_struc_fea/position/test_norm/" + current_name
            seq_file_name = "/home/yjliang/PycharmProjects/structure_bert/generate_data/seq_struc_fea/function/test_norm/" + current_name
            #go_file_name = "/home/yjliang/PycharmProjects/GET_GO_FEA/generate_data/go_my_transformer_fea_norm/position/test/" + current_name
            go_file_name = "/home/yjliang/PycharmProjects/GET_GO_FEA/generate_data/go_my_transformer_fea_norm/function/test/" + current_name

        if label_file_name:
            other_data = sio.loadmat(label_file_name)
            label = other_data['label']
            #label=np.tile(label,(2,1))

        if seq_file_name:
            other_data = sio.loadmat(seq_file_name)
            seq = other_data['final_fea']#'x_pool','final_fea'

        if go_file_name:
            other_data = sio.loadmat(go_file_name)
            go = other_data['final_fea']#go_my_transformer_fea,'final_fea'


        return images,seq, label,go

    def __iter__(self):
        # 在迭代器内部对数据进行随机打乱
        shuffled_indices = torch.randperm(len(self.file_list))
        shuffled_files = [self.file_list[i] for i in shuffled_indices]

        for file in shuffled_files:
            images,seq, label,go = self.parse_file(file)
            yield images,seq, label ,go # 返回特征和标签

    def __len__(self):
        return len(self.file_list)


class MyIterableDataset_test(IterableDataset):
    def __init__(self, file_list,dataset_name):
        super(MyIterableDataset_test, self).__init__()
        self.file_list = file_list
        self.dataset_name=dataset_name


    def parse_file(self, file):
        file_list = "/home/yjliang/PycharmProjects/CLIP/generate_data/final_beat_clip_fea/mean_fea/"+self.dataset_name+"/"+file
        data = sio.loadmat(file_list)
        images = data['final_fea']



        ##还是先要判断是验证集还是训练集
        keyword1 = "train"
        keyword2 = "valid"
        keyword3 = "test"

        if keyword1 in file_list:
            current_name = file
            #label_file_name = "/home/yjliang/PycharmProjects/CLIP/need_data/_5407_divide_dataset_label/train/" + current_name
            label_file_name = "/home/yjliang/PycharmProjects/CLIP/need_data/_5407_divide_dataset_function_label/train/" + current_name
            #seq_file_name = "/home/yjliang/PycharmProjects/structure_bert/generate_data/seq_struc_fea/position/train_norm/" + current_name
            #go_file_name = "/home/yjliang/PycharmProjects/GET_GO_FEA/generate_data/go_my_transformer_fea_norm/position/train/" + current_name
            seq_file_name = "/home/yjliang/PycharmProjects/structure_bert/generate_data/seq_struc_fea/function/train_norm/" + current_name
            go_file_name ="/home/yjliang/PycharmProjects/GET_GO_FEA/generate_data/go_my_transformer_fea_norm/function/train/"+ current_name

        elif keyword2 in file_list:
            current_name = file
            #label_file_name = "/home/yjliang/PycharmProjects/CLIP/need_data/_5407_divide_dataset_label/valid/" + current_name
            label_file_name = "/home/yjliang/PycharmProjects/CLIP/need_data/_5407_divide_dataset_function_label/valid/" + current_name
            #seq_file_name = "/home/yjliang/PycharmProjects/structure_bert/generate_data/seq_struc_fea/position/valid_norm/" + current_name
            seq_file_name = "/home/yjliang/PycharmProjects/structure_bert/generate_data/seq_struc_fea/function/valid_norm/" + current_name
            #go_file_name = "/home/yjliang/PycharmProjects/GET_GO_FEA/generate_data/go_my_transformer_fea_norm/position/valid/" + current_name
            go_file_name = "/home/yjliang/PycharmProjects/GET_GO_FEA/generate_data/go_my_transformer_fea_norm/function/valid/" + current_name

        elif keyword3 in file_list:
            current_name = file
            #label_file_name = "/home/yjliang/PycharmProjects/CLIP/need_data/_5407_divide_dataset_label/test/" + current_name
            label_file_name = "/home/yjliang/PycharmProjects/CLIP/need_data/_5407_divide_dataset_function_label/test/" + current_name
            #seq_file_name = "/home/yjliang/PycharmProjects/structure_bert/generate_data/seq_struc_fea/position/test_norm/" + current_name
            seq_file_name = "/home/yjliang/PycharmProjects/structure_bert/generate_data/seq_struc_fea/function/test_norm/" + current_name
            #go_file_name = "/home/yjliang/PycharmProjects/GET_GO_FEA/generate_data/go_my_transformer_fea_norm/position/test/" + current_name
            go_file_name = "/home/yjliang/PycharmProjects/GET_GO_FEA/generate_data/go_my_transformer_fea_norm/function/test/" + current_name


        if label_file_name:
            other_data = sio.loadmat(label_file_name)
            label = other_data['label']

        if seq_file_name:
            other_data = sio.loadmat(seq_file_name)
            seq = other_data['final_fea']

        if go_file_name:
            other_data = sio.loadmat(go_file_name)
            go = other_data['final_fea']


        return images,seq, label,go,current_name

    def __iter__(self):
        shuffled_indices = torch.randperm(len(self.file_list))
        shuffled_files = [self.file_list[i] for i in shuffled_indices]

        for file in shuffled_files:
            images,seq, label,go,name = self.parse_file(file)
            yield images,seq, label,go,name # 返回特征和标签

    def __len__(self):
        return len(self.file_list)


def main(i):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LR=5*1e-4
    min_loss_val=10000000000000
    batch_size=512
    dim=128

    num_heads =4
    num_layers = 3
    num_classes = 5
    dim_feedforward=512
    dropout=0.0

    # 创建模型
    net = TransformerEncoder(num_layers,dim, num_heads, dim_feedforward, dropout,num_classes).to(device)
    # 定义损失函数和优化器
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    m1 = nn.Sigmoid()


    EPOCH = 100
    Epoch = []
    epoch_losses_train = []
    epoch_losses_valid = []
    net = net.to(device)

    train_file_list="/home/yjliang/PycharmProjects/CLIP/generate_data/final_beat_clip_fea/mean_fea/train/"
    items_train = os.listdir(train_file_list)
    #items_train = items_train[0:10]
    train_data = MyIterableDataset(items_train,'train')
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=False)

    valid_file_list="/home/yjliang/PycharmProjects/CLIP/generate_data/final_beat_clip_fea/mean_fea/valid/"
    items_valid = os.listdir(valid_file_list)
    #items_valid = items_valid[0:10]
    valid_data = MyIterableDataset(items_valid,'valid')
    valid_loader = DataLoader(valid_data, batch_size=batch_size, drop_last=False)

    test_file_list="/home/yjliang/PycharmProjects/CLIP/generate_data/final_beat_clip_fea/mean_fea/test/"
    items_test = os.listdir(test_file_list)
    test_data = MyIterableDataset_test(items_test,'test')
    test_loader = DataLoader(test_data, batch_size=batch_size, drop_last=False)

    for epoch in range(EPOCH):
        Epoch.append(epoch)
        epoch_loss_train = 0
        net.train()

        for images,seq ,label,go in tqdm(train_loader):
            images = torch.tensor(images, dtype=torch.float32)
            seq = torch.as_tensor(seq, dtype=torch.float32)
            combine_fea = torch.cat((images, seq), dim=1)
            go = torch.tensor(go, dtype=torch.float32)
            #output,weight = net(go.to(device), combine_fea.to(device))  # ALL MODE
            output, weight = net(go.to(device), seq.to(device))
            label = label.float()
            label = label.squeeze()
            loss = loss_func(m1(output).to(device), label.to(device))
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            epoch_loss_train += loss.detach().item()
            #print(weight)

        epoch_losses_train.append(epoch_loss_train)

        ##保存模型
        checkpoint = {
            'net': net.state_dict(),  # 保存模型
            'optimizer': optimizer.state_dict(),  # 保存优化器
            'epoch': epoch  # 保存训练轮数
        }

        ##训练完之后，在验证集上跑，计算loss
        net.eval()
        epoch_loss_valid = 0
        with torch.no_grad():
            for images, seq, label ,go in tqdm(valid_loader):
                images = torch.tensor(images, dtype=torch.float32)
                seq = torch.as_tensor(seq, dtype=torch.float32)
                combine_fea = torch.cat((images, seq), dim=1)
                go = torch.tensor(go, dtype=torch.float32)
                #output,weight = net(go.to(device), combine_fea.to(device))
                output, weight = net(go.to(device), seq.to(device))
                label = label.float()
                label = label.squeeze()
                loss = loss_func(m1(output).to(device), label.to(device))
                epoch_loss_valid += loss.detach().item()

            # epoch_loss_valid /= (len(all_file_list_valid))/10000
            # epoch_loss_valid /= 10000
            epoch_losses_valid.append(epoch_loss_valid)
            print('Epoch {}, train_loss {:.4f}'.format(epoch, epoch_loss_train),
                  'valid_loss {:.4f} '.format(epoch_loss_valid))

            ##使用最好的模型进行测试集
            if epoch_loss_valid <= min_loss_val:
                min_loss_val = epoch_loss_valid
                best_model = copy.deepcopy(net)
                pkl_path = '/home/yjliang/PycharmProjects/multi_model_transformer/generate_data/net/best_transformer_mlp_all_mean_value_best'+str(i)+'.pkl'
                torch.save(checkpoint, pkl_path)
                model_test = best_model

    ##画图
    plt.subplot(2, 1, 1)
    plt.plot(Epoch, epoch_losses_train, 'o-')
    plt.title('Train Loss vs. Epoches')
    plt.ylabel('Train Loss')
    plt.subplot(2, 1, 2)
    plt.plot(Epoch, epoch_losses_valid, '.-')
    plt.xlabel('Valid Loss vs. Epoches')
    plt.ylabel('Valid Loss')
    plt.show()

    ##然后是测试集
    model_test.eval()
    Name=[]
    with torch.no_grad():#测试集一定要注意model_test.
        for step, test_data in enumerate(test_loader):
            images, seq, label ,go,name=test_data[:]
            images = torch.tensor(images, dtype=torch.float32)
            seq = torch.as_tensor(seq, dtype=torch.float32)
            combine_fea = torch.cat((images, seq), dim=1)
            go = torch.tensor(go, dtype=torch.float32)
            #output,weight = model_test(go.to(device), combine_fea.to(device))
            output, weight = net(go.to(device), seq.to(device))
            label = label.squeeze()
            mid_Y = m1(output)
            #Name.append(name)
            Name.append([''.join(n) for n in name])


            if step == 0:
                test_probs_Y = mid_Y
                test_Y = label
                Weight = weight.squeeze()
            else:
                test_probs_Y = torch.cat([test_probs_Y, mid_Y], dim=0)
                test_Y = torch.cat([test_Y, label], dim=0)
                Weight = torch.cat([Weight, weight.squeeze()], dim=0)

        print(test_probs_Y)
        print(test_probs_Y.shape)
        print(test_Y.shape)

        # 保存为.mat,画混淆矩阵
        test_Y = np.asarray(test_Y.cpu().detach(), dtype=np.float32)
        test_probs_Y = np.asarray(test_probs_Y.cpu().detach(), dtype=np.float32)
        Weight = np.asarray(Weight.cpu().detach(), dtype=np.float32)
        flattened_name = [str(item) for sublist in Name for item in sublist]


        file_name = "/home/yjliang/PycharmProjects/multi_model_transformer/generate_data/result/result0_mlp_all_mean_value_best"+str(i)+".mat"
        savemat(file_name, {'test_probs_Y': test_probs_Y, 'test_Y': test_Y,'Weight':Weight,'flattened_name':flattened_name})
        print(Weight.shape)


if __name__ == '__main__':
    for i in range(5):
        main(i)
