import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torchvision
import torchtext

import pandas as pd
import numpy as np
from torchtext import data, datasets
import torch.nn as nn
import torch.nn.functional as F


import sys
import csv



f = open(r"C:\Users\Strawberry\Desktop\ml_courses\lhy_lab4_rnn\ml2020spring-hw4\training_label.txt",encoding='utf-8')
lines = []
while True:
    line = f.readline()
    if line:
        if line[-1] == '\n':
            line = line[:-1]
        line = line.split('+++$+++')
        lines.append(line)
    else:
        break

total_nums = len(lines)
div_pos = int(total_nums*3.0/4.0)
# print(total_nums, div_pos)
train_lines = lines[0: div_pos]
test_lines = lines[div_pos: ]
# print(len(train_lines),len(test_lines))

with open('train.csv','w',newline='',encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    for line in train_lines:
        writer.writerow(line)

with open('test.csv','w',newline='',encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    for line in train_lines:
        writer.writerow(line)

TEXT = data.Field(lower=True, batch_first=True, fix_length=50)
LABEL = data.Field(sequential=False)
train, test = data.TabularDataset.splits(path='./', train='train.csv', test='test.csv', 
                format='csv', fields=[('Label',LABEL), ('Text',TEXT)])

# 修改D:\Anaconda\envs\ml_dup1\lib\site-packages\torchtext\utils.py文件第130行
# avoid OVERFLOW ERROR


TEXT.build_vocab(train, vectors="glove.6B.100d", max_size=10000, min_freq=10)
# 这里vectors我们直接放在.\vector_cache文件夹里面
# 预训练词向量加载
LABEL.build_vocab(train)
# print(TEXT.vocab.freqs.most_common(20))
'''
print(len(lines))
print(lines[-1],lines[-2])
'''
mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(mydevice)
train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=16, device=mydevice, shuffle=True, sort=False)
# 创建批处理的迭代器


batch = next(iter(train_iter))
print(batch)
print('batch.Text = \n',batch.Text)
print('batch.Label = \n',batch.Label)




class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, lstm_output_dim, output_dim, 
                num_layers, bidirectional, drop_out, pad_idx, batch_first = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, 
                        padding_idx = pad_idx)
        
        '''
        padding_idx用于batch中非定长seq的处理，将pad的单词映射成0向量
        padding_idx应该为word_bag中padding_word的idx
        '''
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                            batch_first = batch_first, bidirectional=bidirectional,
                            dropout=drop_out)
        '''
        nn.LSTM(input_size, hidden_size, num_layers)
        input_size:输入的feature的维度
        hidden_size:hidden_layer的维度
        num_layers:LSTM的层数？---有待研究
        reference:https://blog.csdn.net/baidu_38963740/article/details/117197619?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link
        '''
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        self.dropout = nn.Dropout(drop_out)
    
    def forward(self, x):
        embedded = self.embedding(x)
        '''
        embedded output size = (batch_size, seq_len, embedding_size)
        according to our former result
        when parameter batch_first = false(default)
        LSTM-input_size are supposed to be : (seq_len, batch_size, embedding_size)
        so we should reverse dim[0] and dim[1] of embedded
        or we set batch_first = True
        '''
        lstm_output, (h_n, c_n) = self.lstm(embedded)
        '''
        when num_layers = bidirectional = 1 and batch_first = True
        size of lstm_output: (batch_size, seq_len, hidden_dim)
        size of h_n and c_n: (num_layers * num_directions = 1,
        batch_size, hidden_size) 
        now we are goin to use state variable h_n rather than output
        (why ?)
        '''
        # processed_h = self.dropout(h_n.squeeze(0))
        # processed_lstmoutput = self.dropout(lstm_output[:, -1, :])
        # output = self.fc(processed_h)
        output = self.dropout(self.fc(lstm_output[:, -1, :]))
        '''
        we only select last_dim for dim_seq_len of lstm_output
        refs:https://zhuanlan.zhihu.com/p/79064602?ivk_sa=1024320u
        so sometimes we can also use h_n
        (similar to last_dim for dim_seq_len of lstm_output)
        '''
        '''
        processed_h:(batch_size, hidden_size)
        output:(batch_size, output_dim(here we supposed to be 2))
        '''
        return F.log_softmax(output, dim = 1)

'''
LSTM--version--log
version1 : 
num_layers = bidirectional = 1 and batch_first = True
And we use the h_n as output
acc on validation is 68%
version2 :
num_layers = bidirectional = 1 and batch_first = True
And we use the lstm_output[:, -1, :] as output
acc on validation is also around 68%
version3 :
change fix_length from 20 to 40 
acc on validation increases to 73%
buttt this is definitely not what I wannt !!!
version4 :
fix_length = 20
add drop_out layer after fc but no works

version5:
fix_length = 200
no works

version6 :
!!! num_layers -> 2 work, acc on validation set reach 83.3%

version7 :
when setting drop_out_rate to 0.8, model behaves bad.
'''

pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
model3 = LSTM(len(TEXT.vocab.stoi), 100, 128, 128, 2, 2, False, 0.5, pad_idx, True)
model3.embedding.weight.data = TEXT.vocab.vectors
model3.embedding.weight.requires_grad = False
model3 = model3.cuda()
'''
(self, vocab_size, embedding_dim, hidden_dim, lstm_output_dim, output_dim, 
num_layers, bidirectional, drop_out, pad_idx, batch_first = False)
'''
optimizer3 = torch.optim.Adam(model3.parameters(),lr=0.001)
def fit3(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
    # model.train()自动开启 batch_normalization 和 drop_out
    # model.eval()沿用 batch_normalization, 不启用 drop_out
    running_loss = 0.0
    running_correct = 0.0
    for batch_idx, batch in enumerate(data_loader):
        text, target = batch.Text, batch.Label
        if mydevice == 'cuda':
            text, target = text.cuda(),target.cuda()
        if phase == 'training':
            optimizer3.zero_grad()
        output = model(text)
        # print(output.shape)
        loss = F.nll_loss(output, target-1)
        running_loss = F.nll_loss(output, target-1, size_average=False).data
        preds = output.data.max(dim=1, keepdim=True)[1] + 1
        # preds返回(batch_size,1),其中每个元素都是最大的下标+1
        # 因为label全是1，2 而非 0，1
        running_correct += preds.eq(target.data.view_as(preds)).sum()
        if phase == 'training':
            loss.backward()
            optimizer3.step()
            # print(model2.embedding.weight.data)
    
    running_loss = running_loss.type(torch.FloatTensor)
    running_correct = running_correct.type(torch.FloatTensor)
    
    # IMPORTANT above! otherwise accuracy will be zero all the time!
    loss = running_loss/len(data_loader.dataset)
    
    accuracy = running_correct/len(data_loader.dataset)
    # print(type(loss),type(accuracy))
    
 
    print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)} {accuracy:{10}.{4}}')
    return loss,accuracy

train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []

train_iter.repeat = False
test_iter.repeat = False

epoch_max3 = 10
for epoch in range(1,epoch_max3):
    epoch_loss, epoch_accuracy = fit3(epoch, model3, train_iter, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit3(epoch, model3, test_iter, phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)



'''
validation acc on twitter dataset reaches around 85%
we use pretrained word_vec(glove) models and LSTM coorperation.
'''









