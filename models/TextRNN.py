# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding, use_premodel):
        self.model_name = 'TextRNN'
        self.use_premodel = use_premodel  # 是否采用预训练模型

        # self.train_path = dataset + '/data/train_test.txt'  # debug用测试集
        # self.dev_path = dataset + '/data/train_test.txt'  # 验证集
        # self.test_path = dataset + '/data/train_test.txt'  # 测试集
        #
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集

        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.gather_class_list = [x.strip() for x in open(
            dataset + '/data/gather_class.txt', encoding='utf-8').readlines()]  # 大类别名单
        # self.index_map_path = dataset + '/data/index_map.txt'  # index_map地址
        # for x in open(self.index_map_path, encoding='utf-8').readlines():  # index_map字典
        #     self.index_map = {}
        #     x = x.strip().split('\t')
        #     self.index_map[x[0]] = x[1:]
        self.vocab_path = dataset + '/data/vocab.pkl'  # 词表
        # self.class_map_path = dataset + '/data/class_map.txt'  # 大分类和小分类映射表
        self.feature_map_path = './84-85-90/rnn/loss_record'  # 大税号要素特征表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.gather_num_classes = len(self.gather_class_list)  # 大税号类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 10  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 128  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数
        self.num_filters = 250  # 卷积核数量(channels数)
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸  # textcnn


'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm1 = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                             bidirectional=True, batch_first=True, dropout=config.dropout)
        self.lstm2 = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                             bidirectional=True, batch_first=True, dropout=config.dropout)

        self.dropout = nn.Dropout(config.dropout)

        # dpcnn卷积层
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embed), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()

        # 结构化卷积
        self.conv2_1 = nn.Conv2d(1, config.num_filters, (2, config.embed))
        self.conv2_2 = nn.Conv2d(1, config.num_filters, (3, config.embed))
        self.conv2_3 = nn.Conv2d(1, config.num_filters, (4, config.embed))

        # self.fc1 = nn.Linear(config.hidden_size*4, config.gather_num_classes)  # 普通bilstm
        self.fc1 = nn.Linear(1256, config.gather_num_classes)  # 普通bilstm
        self.fc2 = nn.Linear(config.gather_num_classes, config.num_classes)

    # def forward(self, x):  # 普通bilstm
    #     x, seq_len1, seq_len2_1, seq_len2_2, seq_len2_3 = x
    #     x1, x2_1, x2_2, x2_3, x3 = x.split(128, dim=1)  # x1要素名称，2_1 2_2 2_3三种卷积核, x3feature
    #
    #     embedded = self.embedding(x2_1)
    #     output, (hidden, cell) = self.lstm1(embedded)
    #     hidden2 = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
    #
    #     embedded = self.embedding(x1)
    #     output, (hidden, cell) = self.lstm2(embedded)
    #     hidden1 = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
    #
    #     hidden = torch.cat((hidden1.squeeze(0), hidden2.squeeze(0)), 1)
    #
    #     z = self.fc1(hidden)
    #     y = self.fc2(z)
    #     return y, z

    def forward(self, x):  # sscnn + bilstm
        x, seq_len1, seq_len2_1, seq_len2_2, seq_len2_3 = x
        x1, x2_1, x2_2, x2_3, x3 = x.split(128, dim=1)  # x1要素名称，2_1 2_2 2_3三种卷积核, x3feature

        x1 = self.embedding(x1).unsqueeze(1)
        x2_1 = self.embedding(x2_1).unsqueeze(1)
        x2_2 = self.embedding(x2_2).unsqueeze(1)
        x2_3 = self.embedding(x2_3).unsqueeze(1)

        x1 = self.conv_region(x1)  # [batch_size, 250, seq_len-3+1, 1]
        x1 = self.padding1(x1)  # [batch_size, 250, seq_len, 1]
        x1 = self.relu(x1)
        x1 = self.conv(x1)  # [batch_size, 250, seq_len-3+1, 1]
        x1 = self.padding1(x1)  # [batch_size, 250, seq_len, 1]
        x1 = self.relu(x1)
        x1 = self.conv(x1)  # [batch_size, 250, seq_len-3+1, 1]
        while x1.size()[2] > 2:
            x1 = self._block(x1)
        x1 = x1.squeeze()  # [batch_size, num_filters(250)]

        x2_1 = self.conv2_1(x2_1)  # [batch_size, 250, seq_len-2+1, 1]
        x2_2 = self.conv2_2(x2_2)  # [batch_size, 250, seq_len-3+1, 1]
        x2_3 = self.conv2_3(x2_3)  # [batch_size, 250, seq_len-4+1, 1]
        while x2_1.size()[2] > 2:
            x2_1 = self._block(x2_1)
        x2_1 = x2_1.squeeze()  # [batch_size, num_filters(250)]
        while x2_2.size()[2] > 2:
            x2_2 = self._block(x2_2)
        x2_2 = x2_2.squeeze()  # [batch_size, num_filters(250)]
        while x2_3.size()[2] > 2:
            x2_3 = self._block(x2_3)
        x2_3 = x2_3.squeeze()  # [batch_size, num_filters(250)]

        embedded = self.embedding(x3)
        output, (hidden, cell) = self.lstm1(embedded)
        hidden1 = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        x2 = torch.cat((x1, x2_1, x2_2, x2_3, hidden1.squeeze(0)), 1)  # [batch_size, 3*num_filters(750)]

        z = self.fc1(x2)
        y = self.fc2(z)
        return y, z

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x

    '''变长RNN，效果差不多，甚至还低了点...'''
    # def forward(self, x):
    #     x, seq_len = x
    #     out = self.embedding(x)
    #     _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
    #     _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
    #     out = torch.index_select(out, 0, idx_sort)
    #     seq_len = list(seq_len[idx_sort])
    #     out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
    #     # [batche_size, seq_len, num_directions * hidden_size]
    #     out, (hn, _) = self.lstm(out)
    #     out = torch.cat((hn[2], hn[3]), -1)
    #     # out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    #     out = out.index_select(0, idx_unsort)
    #     out = self.fc(out)
    #     return out
