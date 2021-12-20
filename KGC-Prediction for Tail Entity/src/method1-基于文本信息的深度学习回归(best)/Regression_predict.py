from gensim.models import word2vec
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F



entity_dict={}
relat_dict={}
with open(r"./lab2_dataset/entity_with_text.txt","r") as fr1, open(r"./lab2_dataset/relation_with_text.txt","r") as fr2, open(r"./lab2_dataset/entityandrelation_co.txt","w") as fw:
    for line in fr1.readlines():
        fw.write(line.strip().split('\t')[1])
        fw.write("\n")
    for line in fr2.readlines():
        fw.write(line.strip().split('\t')[1])
        fw.write("\n")



sentences = word2vec.LineSentence(r'./lab2_dataset/entityandrelation_co.txt')
model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, vector_size=100)
pos_inf = 1e9

hit5 = 0
hit1 = 0
total_num = 0
"""
'''
with open(r".\lab2_dataset\entity_with_text.txt","r") as fr1, open(r".\lab2_dataset\relation_with_text.txt","r") as fr2:
    for line in fr1.readlines():
        dscrip = line.strip().split('\t')[1]
        key = line.strip().split('\t')[0]
        dscrip = dscrip.split()
        sum = 0
        for word in dscrip:
            sum += model.wv[word]
        sum = sum/np.linalg.norm(sum)
        entity_dict[key] = sum

    for line in fr2.readlines():
        dscrip = line.strip().split('\t')[1]
        key = line.strip().split('\t')[0]
        dscrip = dscrip.split()
        sum = 0
        for word in dscrip:
            sum += model.wv[word]
        sum = sum/np.linalg.norm(sum)
        relat_dict[key] = sum
'''

with open(r"/data2/home/zhaoyi/Web_info_lab2/lab2_dataset/entity_with_text.txt","r") as fr1, open(r"/data2/home/zhaoyi/Web_info_lab2/lab2_dataset/relation_with_text.txt","r") as fr2:
    for line in fr1.readlines():
        dscrip = line.strip().split('\t')[1]
        key = line.strip().split('\t')[0]
        dscrip = dscrip.split()
        entity_dict[key] = dscrip

    for line in fr2.readlines():
        dscrip = line.strip().split('\t')[1]
        key = line.strip().split('\t')[0]
        dscrip = dscrip.split()
        relat_dict[key] = dscrip










'''
将 text 全部输入TextCNN，输出得到一个实体（或者关系）的文本表示向量
'''
class TextCNN(nn.Module):
    def __init__(self,  embedding_dim, feature_dim, kernel_size=3):
        super().__init__()
        self.cnn1 = nn.Conv1d(embedding_dim, feature_dim, kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size,stride=2)
        self.cnn2 = nn.Conv1d(feature_dim, 1, kernel_size)
        # output:(batch_size,1,sentence_lens_after2convs)
        self.avg = nn.AdaptiveAvgPool1d(30)
        # output:(batch_size,1,10)
        self.fc = nn.Linear(30, 20)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.maxpool(x)
        x = F.relu(x)
        x = self.cnn2(x)
        x = self.avg(x)
        x = F.relu(x)
        x = torch.squeeze(x,1)
        x = F.dropout(x, 0.5)
        x = self.fc(x)
        return x

mynet = TextCNN(100, 50, 3)
optimizer = torch.optim.SGD(mynet.parameters(),lr=0.001)



def sen2vecs(sen_dscrip, dict):
    vecs = []
    for word in sen_dscrip:
        vec = np.array(model.wv[word])
        vecs.append(vec)
    return np.array(vecs)

max_len = 1000
batch_size = 32

def pad2Dvecs(vecs):
    pad_len = max_len - vecs.shape[0]
    vecs = np.pad(vecs,((0,pad_len),(0,0)))
    return vecs

head_train = []
rela_train = []
tail_train = []
train_loader = []
with open(r"/data2/home/zhaoyi/Web_info_lab2/lab2_dataset/train.txt","r") as fr:
    
    head_batch = []
    rela_batch = []
    tail_batch = []
    line_count = 0
    lines = fr.readlines()
    total = len(lines)
    for line in lines:
        '''
        line = "head relation tail"
        '''
        line_count += 1
        if line_count % 100 == 0:
            print(line_count)
        triple = line.strip().split('\t')
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        if head in entity_dict.keys() and relation in relat_dict.keys() and tail in entity_dict.keys():
            head_dscrip = entity_dict[head]
            rela_dscrip = relat_dict[relation]
            tail_dscrip = entity_dict[tail]
            head_vecs = sen2vecs(head_dscrip, entity_dict)
            rela_vecs = sen2vecs(rela_dscrip, relat_dict)
            tail_vecs = sen2vecs(tail_dscrip, entity_dict)
            '''
            vecs_shape:(sen_len, embedding_dim)
            现在问题的关键在于sen的长度不一致，要想办法padding成一致
            假设我们知道最大sen_len = max_len
            '''
            head_vecs = pad2Dvecs(head_vecs).tolist()
            rela_vecs = pad2Dvecs(rela_vecs).tolist()
            tail_vecs = pad2Dvecs(tail_vecs).tolist()
            head_batch.append(head_vecs)
            rela_batch.append(rela_vecs)
            tail_batch.append(tail_vecs)
        
        if len(head_batch) >= 32 or line_count >= total:
            
            train_loader.append((torch.tensor(head_batch),torch.tensor(rela_batch),torch.tensor(tail_batch)))
            head_batch = []
            rela_batch = []
            tail_batch = []
        if line_count >= 1000:
            break


def train(model,optimizer):
    for head_batch,rela_batch,tail_batch in train_loader:
        '''
        head_batch = torch.tensor(head_batch)
        rela_batch = torch.tensor(rela_batch)
        tail_batch = torch.tensor(tail_batch)
        '''
        head_batch = head_batch.transpose(1,2)
        rela_batch = rela_batch.transpose(1,2)
        tail_batch = tail_batch.transpose(1,2)

        optimizer.zero_grad()
        head_vec = model(head_batch)
        rela_vec = model(rela_batch)
        tail_vec = model(tail_batch)
        loss = torch.norm(head_vec + rela_vec - tail_vec)
        loss.backward()
        optimizer.step()
        print("loss = ",loss, "\n")


count1 = 0
count2 = 0
entdict = {}
for key in entity_dict.keys():
    count1 += 1
    print(count1)
    key_dscrip = entity_dict[key]
    key_vecs = sen2vecs(key_dscrip, entity_dict)
    key_vecs = pad2Dvecs(key_vecs).tolist()
    key_ten = torch.tensor(key_vecs).unsqueeze(0).transpose(1,2)
    key_vec = mynet(key_ten)
    entdict[key] = key_vec
    del key_dscrip,key_vecs,key_ten,key_vec
reldict = {}
for key in relat_dict.keys():
    count2 += 1
    print(count2)
    key_dscrip = relat_dict[key]
    key_vecs = sen2vecs(key_dscrip, relat_dict)
    key_vecs = pad2Dvecs(key_vecs).tolist()
    key_ten = torch.tensor(key_vecs).unsqueeze(0).transpose(1,2)
    key_vec = mynet(key_ten)
    reldict[key] = key_vec
    del key_dscrip,key_vecs,key_ten


def predict(model,head,relation):
    cur_min5 = [pos_inf, pos_inf, pos_inf, pos_inf, pos_inf]
    sel_key5 = [0, 0, 0, 0, 0]
    
    head_vec = entdict[head]
    rela_vec = reldict[relation]

    for tail in entity_dict.keys():
        key = tail
        tail_vec = entdict[key]
        dis = torch.norm(head_vec + rela_vec - tail_vec)
        if dis < cur_min5[4]:
            if dis >= cur_min5[3]:
                cur_min5[4] = dis
                sel_key5[4] = key
            else:
                cur_min5[4] = cur_min5[3]
                sel_key5[4] = sel_key5[3]
                if dis >= cur_min5[2]:
                    cur_min5[3] = dis
                    sel_key5[3] = key
                else:
                    cur_min5[3] = cur_min5[2]
                    sel_key5[3] = sel_key5[2]
                    if dis >= cur_min5[1]:
                        cur_min5[2] = dis
                        sel_key5[2] = key
                    else:
                        cur_min5[2] = cur_min5[1]
                        sel_key5[2] = sel_key5[1]
                        if dis >= cur_min5[0]:
                            cur_min5[1] = dis
                            sel_key5[1] = key
                        else:
                            cur_min5[1] = cur_min5[0]
                            sel_key5[1] = sel_key5[0]
                            cur_min5[0] = dis
                            sel_key5[0] = key
    return sel_key5


epochs = 5
for i in range(epochs):
    train(mynet,optimizer)




with open(r"/data2/home/zhaoyi/Web_info_lab2/lab2_dataset/dev.txt","r") as fr:
    for line in fr.readlines():
        total_num += 1
        triple = line.strip().split('\t')
        head = triple[0]
        relation = triple[1]
        tail_target = triple[2]
        if(head not in entity_dict.keys() or relation not in relat_dict.keys()):
            total_num = total_num - 1
        else: 
            tails_predict = predict(mynet, head, relation)
            if tail_target in tails_predict:
                hit5 += 1
                if tail_target == tails_predict[0]:
                    hit1 += 1
        if(total_num%1000 == 0):
            print('hit@1 = ',hit1*1./total_num,'\n','hit@5 = ',hit5*1./total_num)  
print('hit@1 = ',hit1*1./total_num,'\n','hit@5 = ',hit5*1./total_num)     


"""

'''
思路2: 将word2vec之后的句子concatenation在一起,通过网络再得到一个序列然后判断该序列和尾实体的2norm,作为loss
'''
entity_dict2={}
relat_dict2={}
with open(r"./lab2_dataset/entity_with_text.txt","r") as fr1, open(r"./lab2_dataset/relation_with_text.txt","r") as fr2:
    for line in fr1.readlines():
        dscrip = line.strip().split('\t')[1]
        key = line.strip().split('\t')[0]
        dscrip = dscrip.split()
        sum = 0
        for word in dscrip:
            sum += model.wv[word]
        sum = sum/np.linalg.norm(sum)
        entity_dict2[key] = sum

    for line in fr2.readlines():
        dscrip = line.strip().split('\t')[1]
        key = line.strip().split('\t')[0]
        dscrip = dscrip.split()
        sum = 0
        for word in dscrip:
            sum += model.wv[word]
        sum = sum/np.linalg.norm(sum)
        relat_dict2[key] = sum



class TextDNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim0, hidden_dim1, hidden_dim2, hidden_dim3):
        super().__init__()
        self.fc1 = nn.Linear(2*embedding_dim, hidden_dim0)
        self.cnn = nn.Conv1d(1,1,3)
        self.avg = nn.AdaptiveAvgPool1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, embedding_dim)
    
    def forward(self,x):
        #x = F.dropout(self.fc1(x),0.5)
        x = self.fc1(x)
        x = x.unsqueeze(1)
        x = self.avg(self.cnn(x))
        x = x.squeeze(1)
        x = F.relu(x)
        x = self.fc2(x)
        #x = F.dropout(self.fc2(x),0.5)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

model2 = TextDNN(100,300,200,300,400)
optimizer2 = torch.optim.SGD(model2.parameters(),lr=0.001)

head_train = []
rela_train = []
tail_train = []
train_loader = []
train_filter = {}
with open(r"./lab2_dataset/train.txt","r") as fr:
    
    head_batch = []
    rela_batch = []
    tail_batch = []
    line_count = 0
    lines = fr.readlines()
    total = len(lines)
    for line in lines:
        '''
        line = "head relation tail"
        '''
        line_count += 1
        if line_count % 100 == 0:
            print(line_count)
        triple = line.strip().split('\t')
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        if (head, relation) not in train_filter.keys():
            train_filter[(head, relation)] = []
        train_filter[(head, relation)].append(tail)
        if head in entity_dict2.keys() and relation in relat_dict2.keys() and tail in entity_dict2.keys():
            head_dscrip = entity_dict2[head].tolist()
            rela_dscrip = relat_dict2[relation].tolist()
            tail_dscrip = entity_dict2[tail].tolist()
            head_batch.append(head_dscrip)
            rela_batch.append(rela_dscrip)
            tail_batch.append(tail_dscrip)
        
        if len(head_batch) >= 128 or line_count >= total:
            
            train_loader.append((torch.tensor(head_batch),torch.tensor(rela_batch),torch.tensor(tail_batch)))
            head_batch = []
            rela_batch = []
            tail_batch = []
        
        

def train(model,optimizer):
    for head_batch,rela_batch,tail_batch in train_loader:
        '''
        head_batch:size = (batch_size, embedding_dim)
        '''
        head_rela_batch = torch.cat((head_batch,rela_batch),1)

        optimizer.zero_grad()

        pred_vecs = model(head_rela_batch)
        loss = torch.norm(pred_vecs - tail_batch)
        loss.backward()
        optimizer.step()
        print("loss = ",loss, "\n")



train_epochs = 100
for i in range(train_epochs):
    train(model2, optimizer2)
while 1:
    epochs = int(input())
    for i in range(epochs):
        train(model2, optimizer2)
    if epochs == 0:
        break

def predict(model,head,relation):
    cur_min5 = [pos_inf, pos_inf, pos_inf, pos_inf, pos_inf]
    sel_key5 = [0, 0, 0, 0, 0]
    
    head_vec = entity_dict2[head]
    rela_vec = relat_dict2[relation]

    for tail in entity_dict2.keys():
        if (head, relation) in train_filter.keys() and tail in train_filter[(head, relation)]:
            continue
        key = tail
        tail_vec = torch.tensor(entity_dict2[key])
        head_rela_vec = torch.cat((torch.tensor(head_vec),torch.tensor(rela_vec)),0).unsqueeze(0)
        dis = torch.norm(model2(head_rela_vec).squeeze(0) - tail_vec)
        if dis < cur_min5[4]:
            if dis >= cur_min5[3]:
                cur_min5[4] = dis
                sel_key5[4] = key
            else:
                cur_min5[4] = cur_min5[3]
                sel_key5[4] = sel_key5[3]
                if dis >= cur_min5[2]:
                    cur_min5[3] = dis
                    sel_key5[3] = key
                else:
                    cur_min5[3] = cur_min5[2]
                    sel_key5[3] = sel_key5[2]
                    if dis >= cur_min5[1]:
                        cur_min5[2] = dis
                        sel_key5[2] = key
                    else:
                        cur_min5[2] = cur_min5[1]
                        sel_key5[2] = sel_key5[1]
                        if dis >= cur_min5[0]:
                            cur_min5[1] = dis
                            sel_key5[1] = key
                        else:
                            cur_min5[1] = cur_min5[0]
                            sel_key5[1] = sel_key5[0]
                            cur_min5[0] = dis
                            sel_key5[0] = key
    return sel_key5
'''
with open(r"./lab2_dataset/dev.txt","r") as fr:
    print('here')
    for line in fr.readlines():
        total_num += 1
        triple = line.strip().split('\t')
        head = triple[0]
        relation = triple[1]
        tail_target = triple[2]
        if(head not in entity_dict2.keys() or relation not in relat_dict2.keys()):
            total_num = total_num - 1
        else: 
            tails_predict = predict(model2, head, relation)
            if tail_target in tails_predict:
                hit5 += 1
                if tail_target == tails_predict[0]:
                    hit1 += 1
            if(total_num%1000 == 0):
                print('total_num=',total_num)
                print('hit@1 = ',hit1*1./total_num,'\n','hit@5 = ',hit5*1./total_num)  
print('total_num_over:',total_num)
print('hit@1 = ',hit1*1./total_num,'\n','hit@5 = ',hit5*1./total_num)
'''
with open(r"./lab2_dataset/test.txt","r") as fr, open(r"./result_2.txt","w") as fw:
    print('here')
    for line in fr.readlines():
        total_num += 1
        if(total_num <= 10000):
            continue
        triple = line.strip().split('\t')
        head = triple[0]
        relation = triple[1]
        tail_target = triple[2]
        if(head not in entity_dict2.keys() or relation not in relat_dict2.keys()):
            fw.write('0,0,0,0,0\n')
        else: 
            tails_predict = predict(model2, head, relation)
            fw.write(str(tails_predict[0])+','+str(tails_predict[1])+','+str(tails_predict[2])+','+str(tails_predict[3])+','+str(tails_predict[4])+'\n')
            if(total_num%100 == 0):
                print('total_num=',total_num)
