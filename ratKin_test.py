import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
from utils import *
from model import TCN
import numpy as np
import argparse
import os
import random
os.environ['QT_QPA_PLATFORM']='offscreen' #Needed when ssh to avoid display on screen

import IPython as IP

parser = argparse.ArgumentParser(description='Sequence Regression - Rat Kinematic Data')
parser.add_argument('--batch_size', type=int, default=15, metavar='N',
                    help='batch size (default: 15)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: False)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit (default: 30)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-5,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if not torch.cuda.is_available():
    if args.cuda:
        print("WARNING: You do not have a CUDA device, changing to run model without --cuda")
        args.cuda = False

print(args)

############################################################
# IMPORT DATA
data_dir = os.getcwd() + '/data/'
file_name = 'N5_171016_NoObstacles_s_matrices.mat'

## DATA variables:
neural_sig = 'APdat'            # Name of neural data
decoding_sig = 'KINdat'         # Usually: 'EMGdat' / 'KINdat' (!!string)
decoding_labels = 'KINlabels'   # Usually: 'EMGlabels' / 'KINlabels' (!!string) -> Leave as Empty string otherwise
signal = 3                      # EMG/Kinematic column to decode (FCR,FCU,ECR etc.)
train_prop = 0.90               # Percentage of training data
seq_length = 50                 # Num bins to look at before

data = import_data(data_dir, file_name)
neural_dat, dec_dat, dec_label = define_decoding_data(data, neural_sig, decoding_sig, signal, decoding_labels)


############################################################
'''Prepare data for TCN.
Then we split it into train and test data'''

# Prepare data to feed into TCN:
batch_size = args.batch_size

tcn_x, tcn_y = prepare_TCN_data(neural_dat, dec_dat, seq_length, batch_size)

# Reshape as (N, channels, sample_length)
if batch_size == 1: tcn_x = tcn_x.transpose(0,2,1)
else: tcn_x = tcn_x.transpose(0,1,3,2)
# Create torch Float Tensor
tcn_x, tcn_y = torch.from_numpy(tcn_x).float(), torch.from_numpy(tcn_y).float()

# Now split it into train and test data:
train_idx = int(train_prop * tcn_x.shape[0])

x_train = tcn_x[0:train_idx]
y_train = tcn_y[0:train_idx]
x_train, y_train = shuffle_inputs(x_train, y_train)

x_test = tcn_x[train_idx+1 :]
y_test = tcn_y[train_idx+1 :]

##############################################################

##############################################################
n_classes = 1    # Output size is 1 for normal regression
input_channels = neural_dat.shape[1]
epochs = args.epochs
steps = 0

channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize


model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)
if args.cuda:
    print('Using CUDA')
    model.cuda()
    x_train, x_test, y_train, y_test = x_train.cuda(), x_test.cuda(), y_train.cuda(), y_test.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
# ############################################################


def train(ep):
    global steps
    train_loss = []
    train_err = []
    test_err = []
    running_test_err = []
    # criterion = nn.MSELoss()

    model.train()
    for e in xrange(ep):
        for i in xrange(x_train.shape[0]):

            # print('Training')
            # IP.embed()

            data, target = x_train[i], y_train[i]
            if args.cuda: data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)
            if (batch_size == 1): data = data.unsqueeze(0) # TCN works with a 3D tensor

            optimizer.zero_grad()
            output = model(data)  # Run Network

            loss = F.mse_loss(output, target)  # MSE loss. Take sqrt if you want RMS.
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            if (i%100 == 0):
                _, sq_err = predict(x_test, y_test)
                running_test_err.append(sq_err)
                print('VAF:{}'.format(sq_err))
                train_loss.append(loss.data.cpu().numpy()) #Save running loss


        train_err.append(predict(x_train, y_train)[1])
        test_err.append(predict(x_test, y_test)[1])
        print('Train Epoch: {} \tLoss: {}, train VAF: {}, test VAF {}'.format(ep, train_loss[-1], train_err[-1], test_err[-1]))

    return train_loss, train_err, running_test_err, test_err

# Evaluate your model performance on ANY data
def predict(X, Y):
    model.eval()
    pred = []
    for i in xrange(X.shape[0]):

        data, target = X[i], Y[i]
        if args.cuda: data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        if (batch_size == 1): data = data.unsqueeze(0) # TCN works with a 3D tensor

        output = model(data)
        if (batch_size != 1): output = output.squeeze(1)
        pred.append(output.data.cpu().numpy())

    sq_err = get_corr(Y, pred)
    return pred, sq_err

# Evaluate your model performance on TESTING data
def test():
    model.eval()
    pred = []
    for i in xrange(x_test.shape[0]):
        data, target = x_test[i], torch.FloatTensor([y_test[i]])
        if args.cuda: data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        if (batch_size == 1): data = data.unsqueeze(0) # TCN works with a 3D tensor

        output = model(data)
        pred.append(output.data.cpu().numpy()[0,0])

    sq_err = get_corr(y_test, pred)
    return pred, sq_err

# Variance Accounted For Correlation
def get_corr(y_test, y_test_pred):
    y_test = y_test.cpu().numpy()
    y_mean = np.mean(y_test)
    r2 = 1-np.sum((y_test_pred-y_test)**2)/np.sum((y_test-y_mean)**2)
    return r2

def save_model():
    model_name = 'saved_models/'+ file_name[12] + '.mat'
    save_dir = os.getcwd() + '/' + model_name
    torch.save(model.state_dict(), save_dir)  #SAVE


def plot_after_predict(X, Y):
    # pred, sq_err = predict(X, Y)
    plt.figure()
    plt.plot(Y, label = dec_label[0])
    plt.plot(pred, label = 'Predicted Signal')
    plt.xlabel('Bins')
    plt.ylabel('{} height'.format(dec_label[0]))
    plt.legend()
    plt.title('Actual and predicted kinematic signal, VAF = {}'.format(sq_err))
    plt.show(block=False)

def plot_actual_vs_predict(y, y_pred, sq_err=[]):
    # pred, sq_err = predict(X, Y)
    plt.figure()
    plt.plot(y, label = dec_label[0])
    plt.plot(y_pred, label = 'Predicted Signal')
    plt.xlabel('Bins')
    plt.ylabel('{} height'.format(dec_label[0]))
    plt.legend()
    plt.title('Actual and predicted kinematic signal, VAF = {}'.format(sq_err))
    plt.show(block=False)


if __name__ == "__main__":
    for epoch in range(1, epochs+25):
        train(epoch)
        test()
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
