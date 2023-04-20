from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from utils.config_utils import read_args, load_config, Dict2Object


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """
    tain the model and return the training accuracy
    The train function should use training data (consisted of images-label pairs) to update the model parameters.
    The training loss should be record by a file (e.g., .txt file) after each update.
    The training loss and accuracy should be print after each update.
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    model.train()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # data is the input data of the model (image)
        # target is the label of the input data (the number that the image represents)

        # moving the data and target to the device (GPU)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # setting the gradient to zero before the backward pass (to avoid accumulating the gradient from multiple passes)
        output = model(data)  # getting the output of the model
        # getting the loss of the model (negative log likelihood loss)
        loss = F.nll_loss(output, target)
        loss.backward()  # computing the gradient of the loss w.r.t. the parameters
        optimizer.step()  # updating the parameters of the model

        # record training loss and accuracy
        test_loss += loss.item()  # sum up batch loss

        t_correct = output.argmax(dim=1).eq(target).sum().item()
        correct += t_correct

        print('epoch: {}, batch: {}, loss: {:.6f}, acuuracy: {:.6f}'.format(
            epoch, batch_idx, loss.item(), t_correct/len(target)))
        # record the training loss to a file
        with open('train_loss_acc_{}.txt'.format(args.seed), 'a') as f:
            f.write('epoch: {}, batch: {}, loss: {:.6f}, acuuracy: {:.6f}\n'.format(
                epoch, (batch_idx+1), loss.item(), t_correct/len(target)))

    # calculate the average accuracy
    training_acc = correct / len(train_loader.dataset)
    # calculate the average loss
    training_loss = test_loss / len(train_loader.dataset)
    return training_acc, training_loss


def test(args, model, device, test_loader, epoch):
    """
    test the model and return the tesing accuracy
    The test function should use the testing data (consisted of only images).
    The testing loss and accuracy should be record by a file (e.g., .txt file) after each update.
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            loss = F.nll_loss(output, target)
            test_loss += loss.item()  # sum up batch loss
            t_correct = output.argmax(dim=1).eq(target).sum().item()
            correct += t_correct

            # record the testing loss and accuracy to a file
            msg = 'epoch: {}, loss: {:.6f}, acuuracy: {:.6f}\n'.format(
                epoch, loss.item(), t_correct/len(target))
            with open('test_loss_acc_{}.txt'.format(args.seed), 'a') as f:
                f.write(msg)
            print(msg, end='')

    # calculate the average accuracy
    testing_acc = correct / len(test_loader.dataset)
    # calculate the average loss
    testing_loss = test_loss / len(test_loader.dataset)
    return testing_acc, testing_loss


def plot(epoches, performance, title='model performance', xlabel='epoches', ylabel='performance'):
    """
    plot the model peformance
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title(title)
    plt.plot(epoches, performance)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    pass


def run(config):
    # load config
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    # set device
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': config.batch_size}
    test_kwargs = {'batch_size': config.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True,
                              download=True, transform=transform)  # download the training data
    # download the testing data
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader"""
    if use_cuda:
        generator = torch.cuda.manual_seed(config.seed)  # set random seed
    else:
        generator = torch.manual_seed(config.seed)  # set random seed

    train_loader = torch.utils.data.DataLoader(
        dataset1, **train_kwargs, generator=generator)
    test_loader = torch.utils.data.DataLoader(
        dataset2, **test_kwargs, generator=generator)

    model = Net().to(device)  # load model
    optimizer = optim.Adadelta(
        model.parameters(), lr=config.lr)  # load optimizer
    # optimizer is used to update the parameters of the model to minimize the loss function of the model during training process.

    """record the performance"""
    epoches = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        """training"""
        train_acc, train_loss = train(
            config, model, device, train_loader, optimizer, epoch)

        # record
        training_accuracies.append(train_acc)
        training_loss.append(train_loss)

        print('[TRAIN] epoch: {}, loss: {:.6f}, acuuracy: {:.6f}'.format(
            epoch, train_loss, train_acc))

        """testing"""
        test_acc, test_loss = test(config, model, device, test_loader, epoch)
        # record testing loss and accuracy
        testing_accuracies.append(test_acc)
        testing_loss.append(test_loss)

        print('[TEST] loss: {:.6f}, acuuracy: {:.6f}'.format(
            test_loss, test_acc))

        scheduler.step()  # adjust the learning rate of the optimizer
        epoches.append(epoch)

    """plotting training performance with the records"""
    plot(epoches, training_accuracies, 'training accuracy_{}'.format(config.seed) , 'epoches', 'accuracy')
    plot(epoches, training_loss, 'training loss_{}'.format(config.seed), 'epoches', 'loss')

    """plotting testing performance with the records"""
    plot(epoches, testing_accuracies, 'testing accuracy_{}'.format(config.seed), 'epoches', 'accuracy')
    plot(epoches, testing_loss, 'testing loss_{}'.format(config.seed), 'epoches', 'loss')

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn_{}.pt".format(config.seed))


def plot_mean(config, seeds):
    """
    Read the recorded results.
    Plot the mean results after three runs.
    :return:
    """
    # read the recorded results from three runs
    # and plot the mean results
    # data format:
    # train_loss.txt: epoch: {}, batch: {}, loss: {:.6f}, acuuracy: {:.6f}\n
    # test_loss_acc.txt: epoch: {}, loss: {:.6f}, acuuracy: {:.6f}\n

    import numpy as np

    # read the training loss
    train_loss = [[] for i in range(len(seeds))]
    train_acc = [[] for i in range(len(seeds))]
    for i in range(len(seeds)):
        temp_loss = []
        temp_acc = []
        epoch = 1
        with open('train_loss_acc_{}.txt'.format(seeds[i]), 'r') as f:
            for line in f.readlines():
                data = line.split(',')
                current_epoch = int(data[0].split(':')[-1].strip())
                if current_epoch != epoch:
                    epoch += 1
                    # calculate the mean loss and accuracy for each epoch
                    train_loss[i].append(np.mean(temp_loss))
                    train_acc[i].append(np.mean(temp_acc))
                    temp_loss = []
                    temp_acc = []
                else:
                    temp_loss.append(float(data[-2].split(':')[-1].strip()))
                    temp_acc.append(float(data[-1].split(':')[-1].strip()))
            train_loss[i].append(np.mean(temp_loss))
            train_acc[i].append(np.mean(temp_acc))
    # calculate the mean loss
    train_loss = np.mean(train_loss, axis=0)
    train_acc = np.mean(train_acc, axis=0)

    # read the testing loss and accuracy
    test_loss = [[] for i in range(len(seeds))]
    test_acc = [[] for i in range(len(seeds))]
    for i in range(len(seeds)):
        temp_loss = []
        temp_acc = []
        epoch = 1
        with open('test_loss_acc_{}.txt'.format(seeds[i]), 'r') as f:
            for line in f.readlines():
                data = line.split(',')
                current_epoch = int(data[0].split(':')[-1].strip())
                if current_epoch != epoch:
                    epoch += 1
                    # calculate the mean loss and accuracy for each epoch
                    test_loss[i].append(np.mean(temp_loss))
                    test_acc[i].append(np.mean(temp_acc))
                    temp_loss = []
                    temp_acc = []
                else:
                    temp_loss.append(float(data[-2].split(':')[-1].strip()))
                    temp_acc.append(float(data[-1].split(':')[-1].strip()))
            test_loss[i].append(np.mean(temp_loss))
            test_acc[i].append(np.mean(temp_acc))
    # calculate the mean loss and accuracy
    test_loss = np.mean(test_loss, axis=0)
    test_acc = np.mean(test_acc, axis=0)

    # plot the mean results
    plot(np.arange(1, len(train_loss)+1), train_loss, 'training loss_mean', 'epoches', 'loss')
    plot(np.arange(1, len(train_acc)+1), train_acc, 'training accuracy_mean', 'epoches', 'accuracy')
    plot(np.arange(1, len(test_loss)+1), test_loss, 'testing loss_mean', 'epoches', 'loss')
    plot(np.arange(1, len(test_acc)+1), test_acc, 'testing accuracy_mean', 'epoches', 'accuracy')


if __name__ == '__main__':
    import multiprocessing
    import copy
    import os
    if os.path.exists('config/minist.yaml'):
        arg = argparse.Namespace()
        arg.config_file = 'config/minist.yaml'
    else:
        arg = read_args()

    """toad training settings"""
    config = load_config(arg)

    seeds = config.seeds  # random seeds
    configs = []
    for seed in seeds:
        config_ = copy.deepcopy(config)
        config_.seed = seed
        configs.append(config_)

    """train model and record results"""
    process_list = []
    for i in range(len(seeds)):
        process_list.append(multiprocessing.Process(target=run, args=(configs[i],)))
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()

    """plot the mean results"""
    plot_mean(config, seeds)  # plot the mean results for three runs
