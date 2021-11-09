import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np


is_cuda = 1

transformation = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3801,))]
)

train_dataset = datasets.MNIST('data/', train=True, transform=transformation, download=True)
test_dataset = datasets.MNIST('data/', train=False, transform=transformation, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

def plot_img(image):
    image = image.numpy()[0]
    mean = 0.1307
    std = 0.3081
    image = ((mean * image) + std)
    plt.imshow(image, cmap='gray')
    plt.show()
'''
sample_data = next(iter(train_loader))
plot_img(sample_data[0][1])
plot_img(sample_data[0][2])
'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
        # https://blog.csdn.net/kking_edc/article/details/104663305
        # start batch normalization and drop-out
    if phase == 'validation':
        model.eval()
        # follow batch normalization but no drop-out
        # volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data,target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)
        # 这里volatile应该是requires_grad = ?
        # 没搞清为啥需要这个,回头再看看
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        
        loss = F.nll_loss(output, target)
        running_loss += F.nll_loss(output, target, size_average=False).data
        
        # size_average=False 是对这一个batch的loss求和后的结果
        # 否则表示是对一个batch求平均的loss

        # 为啥这里.data[0]?
        # 应该是书上写错了
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()

        if phase == 'training':
            loss.backward()
            optimizer.step()



    running_loss = running_loss.type(torch.FloatTensor)
    running_correct = running_correct.type(torch.FloatTensor)
    # IMPORTANT above! otherwise accuracy will be zero all the time!
    loss = running_loss/len(data_loader.dataset)
    
    accuracy = running_correct/len(data_loader.dataset)
    # print(type(loss),type(accuracy))
    
 
    print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)} {accuracy:{10}.{4}}')
    return loss,accuracy

model = Net()
if is_cuda:
    model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []
for epoch in range(1, 5):
    epoch_loss, epoch_accuracy = fit(epoch, model, train_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, model, test_loader, phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

plt.plot(range(1, len(train_losses)+1), train_losses, 'bo', label = 'training loss')
plt.plot(range(1, len(val_losses)+1), val_losses, 'r', label = 'validation loss')
plt.legend('loss curve')
plt.show()

plt.plot(range(1, len(train_accuracy)+1), train_accuracy, 'bo', label = 'training accuracy')
plt.plot(range(1, len(val_accuracy)+1), val_accuracy, 'r', label = 'validation accuracy')
plt.legend('accuracy curve')
plt.show()
        
