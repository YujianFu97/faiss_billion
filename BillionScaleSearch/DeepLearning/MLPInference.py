"""
Multilayer Perceptron approach to CIFAR-10 Multi-class classification
Using Pytorch and Cross Entropy Loss
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as td
import random, time
import torchvision
import utils

NCluster = 100
Dimension = 128
scale = 1.0

class FullyConnectedNet(nn.Module):
    def __init__(self):

        self.input_size = Dimension
        self.output_size = NCluster
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(self.input_size, int(1024 * scale))
        self.fc2 = nn.Linear(int(1024 * scale), int(512 * scale))
        self.fc3 = nn.Linear(int(512 * scale), int(256 * scale))
        self.fc4 = nn.Linear(int(256 * scale), self.output_size)

        """
        self.fc1 = nn.Linear(self.input_size, 2634)  # equal spacing between in/out variables
        self.fc2 = nn.Linear(2634, 2196)  # equal spacing between in/out variables
        self.fc3 = nn.Linear(2196, 1758)  # equal spacing between in/out variables
        self.fc4 = nn.Linear(1758, 1320)  # equal spacing between in/out variables
        self.fc5 = nn.Linear(1320, 882)  # equal spacing between in/out variables
        self.fc6 = nn.Linear(882, 444)  # equal spacing between in/out variables
        self.fc7 = nn.Linear(444, self.output_size)  # equal spacing between in/out variables
        """
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        """
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.relu(self.fc4(x))

        x = F.relu(self.fc5(x))

        x = F.relu(self.fc6(x))
        """

        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        
        x = self.fc7(x)
        """
        return x

def cifar_loaders(batch_size, shuffle_test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('/home/yujianfu/Desktop/Dataset/DeepLearningDataset/', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('/home/yujianfu/Desktop/Dataset/DeepLearningDataset/', train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


BaseFilePath = '/home/yujianfu/Desktop/Dataset/SIFT1M/SIFT1M_base.fvecs'
QueryFilePath = '/home/yujianfu/Desktop/Dataset/SIFT1M/SIFT1M_query.fvecs'
FolderPath = '/home/yujianfu/Desktop/Dataset/SIFT1M/DLFiles/'


CenFilePath = FolderPath + 'Centroids_' + str(NCluster) + '.npy'
BaseAssignFilePath = FolderPath + 'BaseAssignment_' + str(NCluster) + '.npy'
QueryAssignFilePath = FolderPath + 'QueryAssignment_' + str(NCluster) + '.npy'

def ANNS_train_loader(batch_size, shuffle_test = False):
    ANNSVector = torch.Tensor(utils.fvecs_read(BaseFilePath))
    ANNSLabel = torch.flatten(torch.Tensor(np.load(BaseAssignFilePath).reshape(-1, 1)).int())
    
    train = td.TensorDataset(ANNSVector, ANNSLabel)
    train_loader = td.DataLoader(train, batch_size= batch_size, shuffle=False)
    return train_loader

def ANNS_test_loader(batch_size, shuffle_test = False):
    ANNSVector = torch.Tensor(utils.fvecs_read(QueryFilePath))
    ANNSLabel = torch.flatten(torch.Tensor(np.load(QueryAssignFilePath).reshape(-1, 1)).int())

    test = td.TensorDataset(ANNSVector, ANNSLabel)
    test_loader = td.DataLoader(test, batch_size =  batch_size, shuffle= False)
    return test_loader


batch_size = 64
test_batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = ANNS_train_loader(batch_size)
test_loader = ANNS_test_loader(test_batch_size)
print(train_loader)

#train_loader, _ = cifar_loaders(batch_size)
#_, test_loader = cifar_loaders(test_batch_size)

# classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ----------------------------------------------------------------------------------------------------------------------
# neural net initialization
# ----------------------------------------------------------------------------------------------------------------------

learning_rate = 1e-2
num_epochs = 100

net = FullyConnectedNet()
NumParas = sum(p.numel() for p in net.parameters())

ModelPath = FolderPath + "MLPModel_" + str(NumParas) + ".pth"
net.to(device)

# ----------------------------------------------------------------------------------------------------------------------
# Loss function and optimizer
# ----------------------------------------------------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.1, verbose=True, cooldown=10)

# ----------------------------------------------------------------------------------------------------------------------
# Train the network
# ----------------------------------------------------------------------------------------------------------------------
print('Run Start Time: ', time.ctime())
begin_time = time.time()
filename = 'Results/Result_' + str(learning_rate) + '_MLP_' + str(time.time()) + '.txt'
f = open(filename, 'w')
f.write('Run Start Time: ' + str(time.ctime()))
f.write('Model Scale: %d, Model Size: %d' % (scale, NumParas))
print('Number of parameters in the model: ', NumParas)
print('Learning Rate: %f' % learning_rate)
f.write('Learning Rate\t%f\n' % learning_rate)

max_accuracy = 0
for epoch in range(num_epochs):  # loop over the dataset multiple times
    start_time = time.time()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        labels = labels.type(torch.LongTensor) 
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs.data, 1)
        correct_matrix = (predicted == labels)
        correct += correct_matrix.sum().item()
        total += labels.size(0)

        # Backward pass + Optimize
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print('Epoch[% d/% d]: Loss: %.4f' % (epoch + 1, num_epochs, running_loss / (i + 1)))
    f.write("Epoch\t%d\tLoss\t%f\t" % (epoch + 1, running_loss / (i + 1)))
    end_time = time.time()
    print("Epoch[%d] total time taken: %f" % (epoch+1, end_time - start_time))
    print('Train Accuracy of the network [%d/%d]: %.2f %%' % (correct, total, 100 * correct / total))

    # ------------------------------------------------------------------------------------------------------------------
    # Test the network
    # ------------------------------------------------------------------------------------------------------------------

    class_correct = list(0. for i in range(NCluster))
    class_total = list(0. for i in range(NCluster))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            correct_matrix = (predicted == labels)
            #print('===============================================')
            #print('out ', outputs.data)
            #print('pred ', predicted)
            #print('labels ', labels)
            #print('correct', correct_matrix)

            c = correct_matrix.squeeze()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            total += labels.size(0)
            correct += correct_matrix.sum().item()
        max_accuracy = max(max_accuracy, int(100 * correct / total))
        scheduler.step(max_accuracy)

    print('Accuray of different classes: ')
    for i in range(100):
        print('%d: %.2f [%d/%d] | ' % (i, 100 * class_correct[i] / class_total[i], class_correct[i], class_total[i]), end = '')

        #print('Accuracy of %5s [%d/%d]: %2f %%' % (classes[i], class_correct[i], class_total[i],
        #                                           100 * class_correct[i] / class_total[i]))
        # f.write('Accuracy of %5s [%d/%d]\t%2f %%\n' % (classes[i], class_correct[i], class_total[i],
        #                                              100 * class_correct[i] / class_total[i]))

    print('Accuracy of the network [%d/%d]: %.2f %%' % (correct, total, 100 * correct / total))
    f.write('Accuracy of the network [%d/%d]\t%.2f %%\n' % (correct, total, 100 * correct / total))
    torch.save(net.state_dict(), ModelPath)
print('Finished Training: ', time.ctime())
f.write('Finished Training: ' + str(time.ctime()) + '\n')
run_time = time.time() - begin_time
print('Total Runtime: %f' % run_time)
f.write('Total Runtime\t%f\n' % run_time)