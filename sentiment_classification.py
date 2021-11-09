import torchtext
from torchtext.legacy import data, datasets
# 注意这里的legacy 版本更新
TEXT = data.Field(lower=True, batch_first=True, fix_length=20)
LABEL = data.Field(sequential=False)

train, test = datasets.IMDB.splits(TEXT, LABEL)
# print(vars(train[0]))
TEXT.build_vocab(train, vectors="glove.6B.100d", max_size=10000, min_freq=10)
LABEL.build_vocab(train)
print(TEXT.vocab.freqs.most_common(20))