import os
import csv
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as func
import torch.utils.data as data

from loss import LossFunc

class AssocDataSet(data.Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)

    def __getitem__(self, index):
        X, y = self.X[index], self.y[index]
        return X, y

    def __len__(self):
        return self.X.size(0)


def train(y_train, X_train, y_val, X_val, ld, frq, beta,
          alpha=1.0, rho=0.9, loss_name='L0'):
    LOSS = {
        'L0' : LossFunc(False, False),
        'L1' : LossFunc(True, False), # Pernalty2
        'L2' : LossFunc(False, True), # Pernalty1
        'L3' : LossFunc(True, True) # Pernalty1 + 2
    }
    Loss = LOSS[loss_name]
    Weight = Variable(torch.FloatTensor(0.5 * np.ones(beta.shape)),
                      requires_grad=True)
    frq = Variable(torch.Tensor(frq))
    ld = Variable(torch.Tensor(ld))

    batch_size = 50
    train_dataset = AssocDataSet(X=X_train, y=y_train)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=1)

    val_X = Variable(torch.Tensor(X_val), requires_grad=False)
    val_y = Variable(torch.Tensor(y_val), requires_grad=False)

    opt = torch.optim.Adam([Weight], lr=0.02)
    scheduler = MultiStepLR(opt,
                            milestones=([x * 5 for x in range(1, 25)]
                                        + [200, 300, 400]),
                            gamma=0.83)

    epoch_iterator = tqdm(range(101))
    for epoch in epoch_iterator:
        epoch_losses = []
        for cur_X, cur_y in train_loader:
            opt.zero_grad()
            cur_X = Variable(cur_X, requires_grad=False)
            cur_y = Variable(cur_y, requires_grad=False)
            loss = Loss(cur_X, cur_y, Weight,
                        alpha=alpha, rho=rho, gamma=frq, tau=ld)
            epoch_losses.append(loss.data[0])
            loss.backward()
            opt.step()

        scheduler.step()

        val_loss = Loss(
            val_X, val_y, Weight,
            alpha=alpha, rho=rho, gamma=frq, tau=ld).data[0]
        status = 'Ephch[{}]: loss: {}, val: {}; rho: {}; alpha: {}'.format(
                epoch,
                np.mean(epoch_losses),
                val_loss,
                rho,
                alpha)
        epoch_iterator.set_description(status)

        weight_name = '{}_rho_{}_alpha_{}.npy'.format(
            loss_name, str(rho)[:3], str(alpha)[:3])
        weight_dir = os.path.join('weight', weight_name)
        weight_file = os.path.abspath(os.path.expanduser(weight_dir))
        weight = Weight.data.numpy()
        np.save(weight_file, weight)

    return val_loss

def main():
    #for rho in [0.1 * i for i in range(0, 11)]:
    path = os.path.abspath(
        os.path.expanduser('~/projects/Exp-assoc'))
    np.random.seed(19260817)
    torch.manual_seed(19260817)
    X = np.loadtxt(os.path.join(path, 'new_data_raw.csv'),
                   dtype = 'float', delimiter = ",", skiprows = 1)
    #b = a[:,1:426].astype('int')
    '''
    beita = [0]*1999
    rd1 =1 - 2*np.random.rand(1,18)
    rd2 =1 - 2*np.random.rand(1,12)
    beita1 = beita + rd1[0].tolist() + [0]*991 + rd2[0].tolist() + [0]*2095
    beita1 = np.array(beita1)
    beita1 = np.expand_dims(beita1, axis=1)
    '''

    with open(os.path.join(path, 'plink.ld')) as f:
        ldlist = f.read().split('\n')[1:-1]
        ld = [float(x.split(',')[2]) for x in ldlist]
        ld = np.array(ld).reshape(-1, 1)

    with open(os.path.join(path, 'plink.frq')) as f:
        frqlist = f.read().split('\n')[1:-1]
        frq = [float(x.split()[4]) for x in frqlist]
        frq = np.array(frq).reshape(-1, 1)

    beta = np.loadtxt(os.path.join(path, 'beta.txt'))
    beta = np.expand_dims(beta, axis = 1)

    e = np.random.normal(loc=0, scale=1, size=X.shape[0])
    e = np.expand_dims(e, axis = 1)

    y = np.dot(X, beta) + e

    y_X = np.hstack((y.reshape(-1, 1), X))
    np.random.shuffle(y_X)
    y, X = y_X[:, :1], y_X[:, 1:]

    for loss in ['L0', 'L1', 'L2', 'L3']:
        res = []
        for alpha in [0.1 * x for x in range(1, 11)]:
            for rho in [0.1 * x for x in range(11)]:
                cvloss = 0
                for i in range(5):
                    lowb = i * 400
                    upb = (i + 1) * 400
                    y_train = np.vstack((y[0 : lowb], y[upb : 2000]))
                    X_train = np.vstack((X[0 : lowb], X[upb : 2000]))
                    y_val = y[lowb : upb]
                    X_val = X[lowb : upb]

                    loss = train(y_train, X_train, y_val, X_val,
                                 ld, frq, beta,
                                 alpha=alpha, rho=rho)
                    cvloss += loss
                cvloss = cvloss / 5.0
                res.append([alpha, rho, cvloss])

        np.savetxt('result_{}.txt'.format(loss), np.array(res))

if __name__ == '__main__':
    main()
