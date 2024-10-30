import os
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import IterableDataset, DataLoader
import scipy.io as sio
from tqdm import tqdm
from model import Seq_Struce_Bert
import torch.autograd as autograd


class MyIterableDataset(IterableDataset):
    def __init__(self, file_list,dataset_name):
        super(MyIterableDataset, self).__init__()
        self.file_list = file_list
        self.dataset_name=dataset_name


    def parse_file(self, file):
        file_path1 = "/home/username/Combined_Fea/"+self.dataset_name+"/"+file
        data1 = sio.loadmat(file_path1)
        struc_raw= data1['final_fea']

        Target_num=2699
        if len(struc_raw)<Target_num:
            aa = len(struc_raw)
            padding_length = Target_num - len(struc_raw)
            padding = np.zeros((padding_length,16))
            struc = np.concatenate((struc_raw, padding))
            mask = np.zeros(Target_num, dtype=bool)
            mask[aa:] = True

        else:
            struc=struc_raw
            mask = np.full(len(struc), False, dtype=bool)

        return struc,mask

    def __iter__(self):
        shuffled_indices = torch.randperm(len(self.file_list))
        shuffled_files = [self.file_list[i] for i in shuffled_indices]

        for file in shuffled_files:
            struc,mask = self.parse_file(file)
            yield struc,mask

    def __len__(self):
        return len(self.file_list)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, 'finish_model.pkl')
        self.val_loss_min = val_loss

class MoCo(nn.Module):
    def __init__(self, embed_size,sequence_length, heads, num_layers, num_classes,  device,dropout,
                 dim,K=2052, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.device = device


        self.encoder_q = Seq_Struce_Bert(embed_size,sequence_length, heads, num_layers, num_classes,  device,dropout)
        self.encoder_k = Seq_Struce_Bert(embed_size,sequence_length, heads, num_layers, num_classes,  device,dropout)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr: ptr + batch_size] = keys.t()
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, final_struc, mask,arg,arg1,arg2, test=False):

        if not test:
            data1 = final_struc
            data2 = final_struc
            L1_output_q, CE_output_q, label_q, q, mask_q= self.encoder_q(data1, mask,arg,arg1,arg2)
            q_mean = torch.mean(q, dim=1)
            q_mean = F.normalize(q_mean, dim=1)
        elif test:
            L1_output_q, CE_output_q, label_q, q, mask_q = self.encoder_q(final_struc, mask,arg,arg1,arg2)
            return L1_output_q, CE_output_q, label_q, q, mask_q,0,0


        with torch.no_grad():
            self._momentum_update_key_encoder()
            L1_output_k, CE_output_k, label_k, k, mask_k= self.encoder_k(data2, mask,arg,arg1,arg2)
            k_mean = torch.mean(k, dim=1)
            k_mean= F.normalize(k_mean, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q_mean, k_mean]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q_mean, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1).to(self.device)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        self._dequeue_and_enqueue(k_mean)
        return L1_output_q, CE_output_q, label_q, q, mask_q,logits, labels

def main(arg,arg1,arg2):
    autograd.set_detect_anomaly(True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    LR=1e-4
    min_loss_val=10000000000000
    batch_size=12
    embed_size=256
    sequence_length=2699
    heads = 4
    num_layers = 3
    num_classes = 35
    dropout=0.1
    patience=4
    EPOCH = 200
    Epoch = []
    epoch_losses_train = []
    epoch_losses_valid = []
    early_stopping = EarlyStopping(patience, verbose=True)
    a=1
    b=1
    c=0.01


    net = MoCo(embed_size,sequence_length, heads, num_layers, num_classes,  device,dropout,dim=256).to(device)
    ##Loading the trained model
    #checkpoint = torch.load(
        #"model path")
    #net.load_state_dict(checkpoint['net'])

    loss_func_L1 = nn.L1Loss()
    loss_func_ce=nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)


    train_file_list="/home/username/Combined_Fea/train/"
    items_train = os.listdir(train_file_list)
    train_data = MyIterableDataset(items_train,'train')
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True)


    valid_file_list="/home/username/Combined_Fea/valid/"
    items_valid = os.listdir(valid_file_list)
    valid_data = MyIterableDataset(items_valid,'valid')
    valid_loader = DataLoader(valid_data, batch_size=batch_size, drop_last=False)

    for epoch in range(EPOCH):
        Epoch.append(epoch)
        epoch_loss_train = 0

        net.train()

        for struc,mask in tqdm(train_loader):
            output_L1, output_ce, label, all_fea, mask_ini, logits, labels = net(struc.to(device),
                                                                                 mask.to(device), arg, arg1,
                                                                                 arg2)
            label_L1 = label[:, :15]
            loss_L1 = loss_func_L1(output_L1, label_L1)
            label_ce = label[:, 15].to(torch.int64)
            loss_CE = loss_func_ce(output_ce, label_ce)
            loss_MOCO = loss_func_ce(logits.to(device), labels.to(device))
            loss = a * loss_CE + b * loss_L1 + c * loss_MOCO
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.detach().item()
        epoch_losses_train.append(epoch_loss_train)

        ##save model
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }

        net.eval()
        epoch_loss_valid = 0
        L1_valid=0
        CE_valid=0
        with torch.no_grad():
            for struc, mask in tqdm(valid_loader):
                output_L1, output_ce, label, all_fea, mask_ini, logits, labels= net(struc.to(device),
                                                                                     mask.to(device), arg, arg1, arg2)#, test=True
                label_L1 = label[:, :15]
                loss_L1 = loss_func_L1(output_L1, label_L1)
                label_ce = label[:, 15].to(torch.int64)
                loss_CE = loss_func_ce(output_ce, label_ce)
                loss_MOCO=loss_func_ce(logits.to(device), labels.to(device))
                loss = a * loss_CE + b * loss_L1 + c*loss_MOCO
                epoch_loss_valid += loss.detach().item()
                L1_valid+= loss_L1.detach().item()
                CE_valid+= loss_CE.detach().item()
            epoch_losses_valid.append(epoch_loss_valid)


            print('Epoch {}, train_loss {:.4f}'.format(epoch, epoch_loss_train),
                  'valid_loss {:.4f} '.format(epoch_loss_valid))


            if epoch_loss_valid <= min_loss_val:
                min_loss_val = epoch_loss_valid
                pkl_path = '/home/username/your_folder/net_' + str(arg) + '_'+ str(arg1) + '_'+ str(arg2) +'.pkl'
                torch.save(checkpoint, pkl_path)

            early_stopping(epoch_loss_valid, net)
            if early_stopping.early_stop:
                print("Early stopping")
                break


    plt.subplot(2, 1, 1)
    plt.plot(Epoch, epoch_losses_train, 'o-')
    plt.title('Train Loss vs. Epoches')
    plt.ylabel('Train Loss')
    plt.subplot(2, 1, 2)
    plt.plot(Epoch, epoch_losses_valid, '.-')
    plt.xlabel('Valid Loss vs. Epoches')
    plt.ylabel('Valid Loss')
    plt.show()





if __name__ == '__main__':
    args = [0.15]
    args1=[0.001]
    args2 = [0.1]

    for i in range(len(args)):
        for j in range(len(args1)):
            main(args[i],args1[j],args2[j])
