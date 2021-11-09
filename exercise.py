import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import os


# ysjx = np.array(Image.open(r'C:\Users\Strawberry\Desktop\ml_courses\cnn_exercises\mnist\ysjx.jpg').resize((224,224)))
'''
ysjx_tensor = torch.from_numpy(ysjx)
print(ysjx_tensor.size())
plt.imshow(ysjx)
plt.show()
plt.imshow(ysjx_tensor[:, :, :].numpy())
plt.show()
'''

'''
a = torch.rand(2, 2)
b = torch.rand(2, 2)
print('a = \n', a)
print('b = \n', b)
c = a + b
d = torch.add(a, b)
print('c = \n', c)
print('d = \n', d)
a.add_(5)
print('a = \n', a)
e = a * b
f = a.mul(b)
h = a @ b
# f 不令a改变； g 令a改变
g = a.mul_(b)

print('e = \n',e,'\nf = \n',f,'\ng = \n',g,'\nh = \n',h)
k = a.matmul(b)
k = torch.matmul(a, b)

a = a.cuda()
b = b.cuda()

i = a.matmul(b)
i = i.to("cpu")
print(k - i)
'''
# 读取方式1
# ysjx = np.array(Image.open(r'C:\Users\Strawberry\Desktop\ml_courses\cnn_exercises\mnist\ysjx.jpg').resize((224,224)))
# 读取方式2
# img = np.array(cv2.imread(r'C:\Users\Strawberry\Desktop\ml_courses\cnn_exercises\mnist\ysjx.jpg'))
# print(img.shape)

# past method

'''
x = Variable(torch.ones(2,2), requires_grad = True)
# now method
x = torch.tensor(np.ones((2,2)), requires_grad = True)
y = x.mean()
y.backward()
print(x.data, '\n', x.grad, '\n', y.data, y.grad_fn)
'''

'''

# m = np.array(1,2,3,4)   wrong
n = np.array([[1,2],[3,4]])
p = np.array(((1,2),(3,4)))
# q = np.asarray(1,2,3,4) wrong
r = np.asarray([[1,2],[3,4]])
s = np.asarray(((1,2),(3,4)))
print(n.shape,p.shape,r.shape,s.shape)
# equivalent above 4 methods
dtype = torch.FloatTensor
print(type(n[1][1]))
X = torch.tensor(np.float32(n), requires_grad= True)
# X = torch.tensor(n, requires_grad= True).type(dtype)
# RuntimeError: Only Tensors of floating point dtype can require gradients
X = X.view(4,1)
# change shape of X
print(X.shape)

# random initialization
w = torch.tensor(np.random.rand(2,2), requires_grad= True)
print(w)

'''

'''
myLayer = nn.Linear(in_features=10, out_features=5, bias=True)
# RuntimeError: Expected object of scalar type Float but got scalar type Double for argument #4 'mat1'
# .type(float) is necessary!!!
inp = torch.tensor(np.random.rand(1,10), requires_grad=True).type(torch.FloatTensor)
# inp = Variable(torch.randn(1, 10))
myLayer(inp)
# paras of layer was automatically initialized
print(myLayer.weight,'\n',myLayer.bias)
'''
'''
sample_data = np.array([1,2,-1,-1],dtype=float)
sample_data = torch.tensor(sample_data,requires_grad=True)
# RuntimeError: Only Tensors of floating point dtype can require gradients
myRelu = nn.ReLU()
print(myRelu(sample_data))
# print(Variable(torch.LongTensor(3).random_(5)),torch.randn(3,5))


loss = nn.MSELoss()
loss2 = nn.CrossEntropyLoss()
input = torch.tensor(np.random.rand(3, 5), requires_grad=True)
target = torch.tensor(np.random.randint(0,2,size=3)).type(torch.LongTensor)
# 注意这里size 不能写成(3,1)
# multi-target not supported at C:\w\1\s\tmp_conda_3.7_055457\conda\conda-bld\pytorch_1565416617654\work\aten\src\THNN/generic/ClassNLLCriterion.c:22
# .type(long) is necessary!
# torch.tensor(data, dtype=None, device=None, requires_grad=False)
# output = loss(input, target)

output2 = loss2(input, target)
output2.backward()
print(output2)
print(input.grad)
'''

'''
print(Variable(torch.tensor([1,3,4,2]).type(torch.FloatTensor),False))

a = np.array([[1,2,3],
             [2,3,4]])
print(a,a.shape)
b = torch.tensor(a)
print(type(b),b,type(b.data),b.data)
print(b.max())
'''
'''
x = Variable(torch.tensor([0.2, 0.3]))
w1 = Variable(torch.tensor([0.6, 0.2]),requires_grad=True)
w2 = Variable(torch.tensor([0.1, 0.7]),requires_grad=True)
w3 = Variable(torch.tensor([0.5, 0.8]),requires_grad=True)
x1 = torch.relu(torch.matmul(x, w1))
x2 = torch.relu(torch.matmul(x, w2))
x3 = torch.stack([x1, x2], -1)
print(x3)
output = torch.relu(torch.matmul(x3, w3))
print(output)
loss = 0.5*torch.pow((0.5 - output), 2)
loss.backward()
print(w1.grad, w2.grad, w3.grad)
'''


def forward(x, w1, w2, w3):
    x1 = torch.relu(torch.matmul(x, w1))
    x2 = torch.relu(torch.matmul(x, w2))
    x3 = torch.stack([x1, x2], -1)
    output = torch.relu(torch.matmul(x3, w3))
    print("output = ", output)
    loss = 0.5*torch.pow((0.5 - output), 2)
    return loss

x = Variable(torch.tensor([0.2, 0.3]))
w1 = Variable(torch.tensor([0.6, 0.2]),requires_grad=True)
w2 = Variable(torch.tensor([0.1, 0.7]),requires_grad=True)
w3 = Variable(torch.tensor([0.5, 0.8]),requires_grad=True)
loss = forward(x, w1, w2, w3)
loss.backward()
print(loss)

lr = 1
print(w1.grad, w2.grad, w3.grad)
w1.data = w1.data - lr * w1.grad.data
w2.data = w2.data - lr * w2.grad.data
w3.data = w3.data - lr * w3.grad.data
w1.grad.data.zero_()
w2.grad.data.zero_()
w3.grad.data.zero_()
print(w1, w2, w3)
loss = forward(x, w1, w2, w3)
loss.backward()
print(loss, w1.grad, w2.grad, w3.grad)


