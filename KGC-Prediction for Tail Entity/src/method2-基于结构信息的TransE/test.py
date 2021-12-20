import codecs
import random

import numpy as np


entityId2vec = {}
relationId2vec = {}


def construct_triple_list(test_file):
    with open(test_file,'r') as f:
        lines = f.readlines()
        entities = set()
        relations = set()
        triples = []
        for line in lines:
            line = line.strip().split('\t')
            triples.append([line[0],line[2],line[1]])

            entities.add(line[0])
            entities.add(line[2])
            relations.add(line[1])            
    
    return entities, relations, triples

def get_entityandrelation(train_file):
    with open(test_file,'r') as f:
        lines = f.readlines()
        entities = set()
        relations = set()
        triples = []
        for line in lines:
            line = line.strip().split('\t')
            triples.append([line[0],line[2],line[1]])

            entities.add(line[0])
            entities.add(line[2])
            relations.add(line[1])            
    
    return entities, relations, triples

def transE_loader(root_file):
    file1 = root_file + "entity_50dim1"
    # file1 = file + "entity_50dim"
    file2 = root_file + "relation_50dim1"
    # file2 = file + "relation_50dim"
    with codecs.open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            line = line.strip().split("\t")
            entityId2vec[line[0]] = eval(line[1])
    with codecs.open(file2, 'r') as f:
        content = f.readlines()
        for line in content:
            line = line.strip().split("\t")
            relationId2vec[line[0]] = eval(line[1])


def distance(h, r, t):
    h = np.array(h)
    r = np.array(r)
    t = np.array(t)
    s = h + r - t
    return np.linalg.norm(s)

pos_inf = 1e10
def mean_rank(entity_set, triple_list):
    # triple_batch = random.sample(triple_list, 100)
    triple_batch = triple_list
    mean = 0
    hit5 = 0
    hit1 = 0
    for triple in triple_batch:
        dlist = []
        h = triple[0]
        t = triple[1]
        r = triple[2]
        if(h not in entityId2vec.keys() or r not in relationId2vec.keys() or t not in relationId2vec.keys()):
            dlist.append((t, pos_inf))
        else:
            dlist.append((t, distance(entityId2vec[h], relationId2vec[r], entityId2vec[t])))
        for t_ in entity_set:
            if t_ != t:
                if(h not in entityId2vec.keys() or r not in relationId2vec.keys() or t_ not in relationId2vec.keys()):
                    dlist.append((t_, pos_inf))
                else:
                    dlist.append((t_, distance(entityId2vec[h], relationId2vec[r], entityId2vec[t_])))
        dlist = sorted(dlist, key=lambda val: val[1])
        for index in range(len(dlist)):
            if dlist[index][0] == t:
                mean += index + 1
                if index < 1:
                    hit1 += 1
                if index < 5:
                    hit5 += 1
                print(index)
                break
    print("mean rank:", mean / len(triple_batch))
    print("hit@1:", hit1 / len(triple_batch))
    print("hit@5:", hit5 / len(triple_batch))


if __name__ == '__main__':
    test_file = r"C:\Users\Strawberry\Desktop\Web_info_lab2\lab2_dataset\dev.txt"
    print("load file...")
    entity_set, relation_set, triple_list = construct_triple_list(test_file)
    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))
    print("load transE vec...")
    transE_loader(".\\")
    print("Complete load.")
    mean_rank(entity_set, triple_list)