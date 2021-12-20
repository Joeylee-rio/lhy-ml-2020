'''
reproduce the result of maml_regression of paper MAML
target function y = f(x) = a*sin(x + b)
where: 
amplitude:a varies within [0.1, 5.0]
phase:b varies within [0, pi]
sampled data_points:x uniformly sampled from [-5.0, 5.0]
loss function: MSE
regressor structure: feed-forward nn with 2 hidden layers of size 40 with ReLU nonlinearities
K = 10, alpha = 0.01, meta-optimizer: Adam
'''
import torch
import torch.nn as nn
from torch.optim import optimizer
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mydevice = 'cpu'
'''
生成100组待拟合的sine函数, training task set
'''
task_num = 300
amp_min, amp_max = 0.1, 5.0
phs_min, phs_max = 0, np.pi
K = 10
x_min, x_max = -5.0, 5.0
"""
np.random.seed()
amps = np.random.uniform(low=amp_min, high=amp_max, size=task_num)
phss = np.random.uniform(low=phs_min, high=phs_max, size=task_num)
x_set = []
y_set = []
for t in range(task_num):
    x = np.random.uniform(low=x_min, high=x_max, size=2*K)
    '''
    这里size = 2*K的原因是我们一半(K)的点用来inner_train, 另外一半的点用来
    outer_train
    '''
    x_set.append(x)
    y = amps[t]*np.sin(x + phss[t])
    y_set.append(y)
"""






'''
生成一组测试数据
'''
amp_val = np.random.uniform(low=amp_min, high=amp_max, size=1)
phs_val = np.random.uniform(low=phs_min, high=phs_max, size=1)
x_val = np.random.uniform(low=x_min, high=x_max, size=K)
y_val = amp_val*np.sin(x_val + phs_val)
plot_x_val = np.linspace(-5,5,1000)
plot_y_val = amp_val*np.sin(plot_x_val+phs_val)


'''
将数据转化成dataloader(self-define)
'''
def list_of_groups(init_list, childern_list_len):
    '''
    init_list为初始化的列表，childern_list_len初始化列表中的几个数据组成一个小列表
    :param init_list:
    :param childern_list_len:
    :return:
    '''
    list_of_group = zip(*(iter(init_list),) *childern_list_len)
    end_list = [list(i) for i in list_of_group]
    count = len(init_list) % childern_list_len
    end_list.append(init_list[-count:]) if count !=0 else end_list
    return end_list

'''
print(x_set)
print(y_set)

print(len(plot_x_set))
print(len(plot_y_set))
'''
'''
tasks = []
'''

'''
tasks列表：elemtype=(task_id,amps[task_id],phss[task_id],x_set[task_id],y_set[task_id])
'''
'''
for i in range(task_num):
    tasks.append((i, amps[i], phss[i], x_set[i], y_set[i])) 
    

batch_size = 32
dataloader = list_of_groups(tasks, batch_size)
'''


def gen_data(task_num = 300,batch_size = 32):
    np.random.seed()
    amps = np.random.uniform(low=amp_min, high=amp_max, size=task_num)
    phss = np.random.uniform(low=phs_min, high=phs_max, size=task_num)
    x_set = []
    y_set = []
    for t in range(task_num):
        x = np.random.uniform(low=x_min, high=x_max, size=2*K)
        '''
        这里size = 2*K的原因是我们一半(K)的点用来inner_train, 另外一半的点用来
        outer_train
        '''
        x_set.append(x)
        y = amps[t]*np.sin(x + phss[t])
        y_set.append(y)
    tasks = []
    for i in range(task_num):
        tasks.append((i, amps[i], phss[i], x_set[i], y_set[i]))
    dataloader = list_of_groups(tasks, batch_size)
    return dataloader

class feed_forward_dnn(nn.Module):
    def __init__(self):
        super(feed_forward_dnn, self).__init__()
        self.hidden_layer1 = nn.Linear(1, 40)
        self.hidden_layer2 = nn.Linear(40, 40)
        self.output_layer = nn.Linear(40, 1)

    def forward(self, x):
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        return self.output_layer(x)
    
    def inner_forward(self, x, params):
        '''
        适用于inner_loop的forward计算
        params: OrderedDict(model.named_parameters())
        '''
        x = F.relu(F.linear(x, params[f'hidden_layer1.weight'], params[f'hidden_layer1.bias']))
        x = F.relu(F.linear(x, params[f'hidden_layer2.weight'], params[f'hidden_layer2.bias']))
        return F.linear(x, params[f'output_layer.weight'], params[f'output_layer.bias'])

model = feed_forward_dnn()
meta_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.MSELoss()

pretrain_model = feed_forward_dnn()
pretrain_optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=0.001)

pretrain_model.state_dict()["hidden_layer1.weight"].copy_(model.state_dict()["hidden_layer1.weight"]) 
pretrain_model.state_dict()["hidden_layer2.weight"].copy_(model.state_dict()["hidden_layer2.weight"]) 
pretrain_model.state_dict()["hidden_layer1.bias"].copy_(model.state_dict()["hidden_layer1.bias"]) 
pretrain_model.state_dict()["hidden_layer2.bias"].copy_(model.state_dict()["hidden_layer2.bias"]) 
pretrain_model.state_dict()["output_layer.weight"].copy_(model.state_dict()["output_layer.weight"]) 
pretrain_model.state_dict()["output_layer.bias"].copy_(model.state_dict()["output_layer.bias"]) 
'''
复制网络初始化参数
'''
simple_model = feed_forward_dnn()
simple_optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
def Pretrain_train(model, optimizer, dataloader, loss_func):
    model.train()
    criterion = loss_func
    batch_losses = []
    for batch in dataloader:
        task_loss = []
        for task in batch:
            (id, amp, phs, x, y) = task
            x = x[:K]
            y = y[:K]
            x = torch.tensor(x.reshape((-1,1))).type(torch.FloatTensor)
            y = torch.tensor(y.reshape((-1,1))).type(torch.FloatTensor)
            x_hat = pretrain_model.forward(x)
            loss = criterion(x_hat, y)
            task_loss.append(loss)
        optimizer.zero_grad()
        batch_loss = torch.stack(task_loss).mean()
        batch_loss.backward()
        batch_losses.append(batch_loss)
        optimizer.step()
        
    return torch.stack(batch_losses).mean()

def Pretrain_test(x, y, model, loss_func, optimizer):
    x = torch.tensor(x.reshape((-1,1))).type(torch.FloatTensor)
    y = torch.tensor(y.reshape((-1,1))).type(torch.FloatTensor)
    x_hat = model.forward(x)
    loss = loss_func(x_hat, y)
    model.train()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def Simple_test(x, y, model, loss_func, optimizer):
    x = torch.tensor(x.reshape((-1,1))).type(torch.FloatTensor)
    y = torch.tensor(y.reshape((-1,1))).type(torch.FloatTensor)
    x_hat = model.forward(x)
    loss = loss_func(x_hat, y)
    model.train()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

'''
好像有问题，这里lossfunction不应该这样backward
'''
def universe_test(amp, phs, model, loss_func, optimizer):
    x = np.linspace(-5,5,1000)
    y = amp*np.sin(x+phs)
    x = torch.tensor(x.reshape((-1,1))).type(torch.FloatTensor)
    y = torch.tensor(y.reshape((-1,1))).type(torch.FloatTensor)
    x_hat = model.forward(x)
    loss = loss_func(x_hat, y)
    model.train()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def MAML_train(model, optimizer, dataloader, loss_func, inner_train_step=1, inner_train_lr=0.01):

    criterion = loss_func
    
    batch_losses = []
    for meta_batch in dataloader:
        task_loss = []
        for task in meta_batch:
            '''
            regression任务应该一个一个输入神经网络，不能一个batch的任务一起
            输入神经网络
            '''
            (id, amp, phs, x, y) = task
            x = torch.tensor(x.reshape((-1,1))).type(torch.FloatTensor)
            y = torch.tensor(y.reshape((-1,1))).type(torch.FloatTensor)
            support_x = x[:K]
            query_x = x[K:]
            support_y = y[:K]
            query_y = y[K:]
            fast_weights = OrderedDict(model.named_parameters())
            fast_weights_debug = OrderedDict(model.named_parameters())
            '''
            fast_weights 是对 model_params的一个复制，使得我们可以
            在inner_loop内更新fast_weights，而不改变 model_params的原值
            '''
            for i in range(inner_train_step):
                sup_x_hat = model.inner_forward(support_x, fast_weights)
                loss = criterion(sup_x_hat, support_y)
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True, retain_graph=True)
                '''
                保留torch的计算图，使得可以二次求导， 也就是后面meta_update时 可以backward()
                torch.autograd.gard并不会给paras.grad赋值
                '''
                # print(fast_weights['output_layer.bias'].grad)
                fast_weights = OrderedDict((name, param - inner_train_lr * grad) 
                                        for ((name, param), grad) in zip(fast_weights.items(), grads))
                
                
                '''
                手动更新每个inner_train_step后fast_weights的变化
                '''
                
            '''
            print(fast_weights['output_layer.bias'])
            print(fast_weights_debug['output_layer.bias'])
            我现在大致理解backward()机制了，这是针对地址（address）的反向求导
            fast_weights_debug一直标定的是原optimizier的参数地址
            而fast_weights一直标定的是自己在变化的地址（∇loss 来 update θ 变成 θ'，一开始的初始地址还是fai）
            '''


          
            qry_x_hat = model.inner_forward(query_x, fast_weights)
            
            loss = criterion(qry_x_hat, query_y)
            task_loss.append(loss)
       
        model.train()
        optimizer.zero_grad()
        meta_batch_loss = torch.stack(task_loss).mean()
        meta_batch_loss.backward()
        '''

        '''
        batch_losses.append(meta_batch_loss)
        optimizer.step()

    return torch.stack(batch_losses).mean()



def MAML_test(x, y, model, loss_func, optimizer):
    x = torch.tensor(x.reshape((-1,1))).type(torch.FloatTensor)
    y = torch.tensor(y.reshape((-1,1))).type(torch.FloatTensor)
    x_hat = model.forward(x)
    loss = loss_func(x_hat, y)
    model.train()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def plot(model1, model2, model3, plot_x, plot_y, test_x, test_y):
    plot_x= torch.tensor(plot_x.reshape((-1,1))).type(torch.FloatTensor)
    plot_x_hat1 = model1.forward(plot_x).squeeze().detach().numpy()
    plot_x_hat2 = model2.forward(plot_x).squeeze().detach().numpy()
    plot_x_hat3 = model3.forward(plot_x).squeeze().detach().numpy()
    fig = plt.figure(figsize = [9.6,7.2])
    ax = plt.subplot(111)
    plot_x1 = plot_x.squeeze().numpy()
    ax.scatter(test_x.squeeze(), test_y.squeeze())
    ax.plot(plot_x1, plot_y, label = 'origin')
    ax.plot(plot_x1, plot_x_hat1, label = 'maml')
    ax.plot(plot_x1, plot_x_hat2, label = 'pretrain')
    ax.plot(plot_x1, plot_x_hat3, label = 'simple regression')
    plt.title(f'K={K}, inner_lr={inner_lr}, training_epochs={train_epochs}, testing_epochs={test_epochs}, inner_updates ={inner_updates}, tasknums = {task_num}')
    ax.legend()
    plt.show()
    

train_epochs = 5000
test_epochs = 1

inner_lr = 0.01
inner_updates = 1

# simple regression
for i in range(test_epochs):
    test_loss = Simple_test(x_val,y_val,simple_model,loss_func,simple_optimizer)
    print(f'simple regression:the loss of {i+1}th-epoch-testing is {test_loss:{5}.{3}}')



# pretrain_model
pretrain_loss = []

for i in range(train_epochs):
    dataloader = gen_data(200,32)
    train_loss = Pretrain_train(pretrain_model, pretrain_optimizer, dataloader, 
            loss_func)
    print(f'pretrain:the loss of {i+1}th-epoch-training is {train_loss:{5}.{3}}') 
for i in range(test_epochs):
    '''
    test_loss = universe_test(amp_val,phs_val,pretrain_model,loss_func,pretrain_optimizer)
    '''
    test_loss = Pretrain_test(x_val,y_val,pretrain_model, loss_func,pretrain_optimizer)
    print(f'pretrain:the loss of {i+1}th-epoch-testing is {test_loss:{5}.{3}}')
    pretrain_loss.append(test_loss.detach().numpy())



# maml_model
maml_loss = []
for i in range(train_epochs):
    dataloader = gen_data(200,32)
    train_loss = MAML_train(model, meta_optimizer, dataloader, 
            loss_func, inner_train_step=inner_updates, inner_train_lr=inner_lr)
    print(f'maml:the loss of {i+1}th-epoch-training is {train_loss:{5}.{3}}') 
for i in range(test_epochs):
    # test_loss = universe_test(amp_val,phs_val,model,loss_func,meta_optimizer)
    test_loss = MAML_test(x_val, y_val, model, loss_func, meta_optimizer)
    print(f'maml:the loss of {i+1}th-epoch-testing is {test_loss:{5}.{3}}')
    maml_loss.append(test_loss.detach().numpy())



plot(model, pretrain_model,simple_model, plot_x_val, plot_y_val, x_val, y_val)

for k in range(10):
    test_epochs += 1
    for i in range(1):
        # test_loss = universe_test(amp_val,phs_val,pretrain_model,loss_func,pretrain_optimizer)
        test_loss = Pretrain_test(x_val,y_val,pretrain_model, loss_func,pretrain_optimizer)
        print(f'pretrain:the loss of {i+1}th-epoch-testing is {test_loss:{5}.{3}}')
        pretrain_loss.append(test_loss.detach().numpy())
    for i in range(1):
        # test_loss = universe_test(amp_val,phs_val,model,loss_func,meta_optimizer)
        test_loss = MAML_test(x_val, y_val, model, loss_func, meta_optimizer)
        print(f'maml:the loss of {i+1}th-epoch-testing is {test_loss:{5}.{3}}')
        maml_loss.append(test_loss.detach().numpy())

    plot(model, pretrain_model,simple_model, plot_x_val, plot_y_val, x_val, y_val)



plt.plot(range(1, len(pretrain_loss)+1), pretrain_loss, label = 'pretrain loss')
plt.plot(range(1, len(maml_loss)+1), maml_loss, label = 'maml loss')
plt.legend('loss curve')
plt.show()

