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

import IPython as IP

parser = argparse.ArgumentParser(description='Sequence Regression - Rat Kinematic Data')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size (default: 1)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: False)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit (default: 10)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-3,
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
neural_sig, dec_sig, dec_label = define_decoding_data(data, neural_sig, decoding_sig, signal, decoding_labels)


# Split train & test data
'''TODO: Split train and test data after having prepared it for TCN.
To make cross validation easier and not loose the prev_bins twice'''
train_idx = int(train_prop * neural_sig.shape[0])

train_neural_sig = neural_sig[0:train_idx]
train_dec_sig = dec_sig[0:train_idx]

test_neural_sig = neural_sig[train_idx+1 :]
test_dec_sig = dec_sig[train_idx+1 :]

# Prepare data to feed into TCN
x_train, y_train = prepare_TCN_data(train_neural_sig, train_dec_sig, seq_length)
x_test, y_test = prepare_TCN_data(test_neural_sig, test_dec_sig, seq_length)

# Reshape as (N, channels, sample_length)
x_train, x_test = x_train.transpose(0,2,1), x_test.transpose(0,2,1)

# Create torch Float Tensors
x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train)
x_test, y_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test)

############################################################

# ############################################################
batch_size = args.batch_size
n_classes = 1    # Output size is 1 for normal regression
input_channels = neural_sig.shape[1]
epochs = args.epochs
steps = 0

print(args)

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
    # criterion = nn.MSELoss()
    model.train()
    for i in xrange(x_train.shape[0]):
        data, target = x_train[i], torch.FloatTensor([y_train[i]])
        if args.cuda: data, target = data.cuda(), target.cuda()
        # print('Training')
        # IP.embed()

        data, target = Variable(data), Variable(target)
        if (batch_size == 1): data = data.unsqueeze(0) # TCN works with a 3D tensor

        optimizer.zero_grad()
        output = model(data)  # Run Network

        # print('Loss')
        # IP.embed()
        loss = F.mse_loss(output, target)  # MSE loss. Take sqrt if you want RMS.
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        if (i%100 == 0):
            _, sq_err = test()
            print('Error:{}'.format(sq_err))

    train_loss.append(loss)
    print('Train Epoch: {} \tLoss: {}'.format(ep, train_loss[-1]))


def test():
    model.eval()
    pred = []
    for i in xrange(x_test.shape[0]):
        data, target = x_test[i], torch.FloatTensor([y_test[i]])
        if args.cuda: data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        if (batch_size == 1): data = data.unsqueeze(0) # TCN works with a 3D tensor


        # IP.embed()
        output = model(data)

        # test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred.append(output.data.numpy()[0,0])

    sq_err = np.sum(np.sqrt((y_test.numpy() - pred)**2))
    return pred, sq_err

    # test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    # return test_loss


if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        train(epoch)
        test()
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
