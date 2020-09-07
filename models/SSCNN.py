# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding, use_premodel):
        self.use_premodel = use_premodel  # 是否采用预训练模型
        self.model_name = 'DPCNN'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        # self.train_path = dataset + '/data/train_test.txt'  # debug用测试集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        # self.dev_path = dataset + '/data/train_test.txt'  # 验证集
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
        self.feature_map_path = './84-85-90/cnn/loss_record'  # 大税号要素特征表
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
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 128  # 每句话处理成的长度(短填长切)
        # self.feature_size = 32  # 重要特征处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度
        self.num_filters = 250  # 卷积核数量(channels数)
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸  # textcnn
        self.a = 0.5  # 扩大倍数
        self.dim_model = 300
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5
        self.num_encoder = 2


'''Deep Pyramid Convolutional Neural Networks for Text Categorization'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

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

        # textcnn
        self.convs_add = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])

        # attention
        # self.tanh1 = nn.Tanh()
        # self.w = nn.Parameter(torch.Tensor(config.num_filters))
        # self.tanh2 = nn.Tanh()

        # for p in self.parameters():  # 冻结上面层的参数
        #     p.requires_grad = False

        # 两个全连接层
        self.fc1 = nn.Linear(config.num_filters * 10, config.gather_num_classes)
        # self.fc1 = nn.Linear(config.pad_size * config.dim_model + 1000, config.gather_num_classes) # transform
        # self.fc1 = nn.Linear(config.num_filters, config.gather_num_classes)
        self.fc2 = nn.Linear(config.gather_num_classes, config.num_classes)

        # transform
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout, config.device)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(config.num_encoder)])

    def forward(self, x):
        """
        # 主cnn 副transform
        x = x[0]
        x = self.embedding(x)
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embed, 1]
        # x1, x2, x3 = x.split(128, dim=2)  # x1要素名称，x2要素内容, x3重要特征
        x1, x2 = x.split(64, dim=2)  # x1要素名称，x2要素内容, x3重要特征
        # 共享卷积 x1 要素名称
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
        # x2 要素内容
        x2 = self.conv_region(x2)  # [batch_size, 250, seq_len-3+1, 1]
        x2 = self.padding1(x2)  # [batch_size, 250, seq_len, 1]
        x2 = self.relu(x2)
        x2 = self.conv(x2)  # [batch_size, 250, seq_len-3+1, 1]
        x2 = self.padding1(x2)  # [batch_size, 250, seq_len, 1]
        x2 = self.relu(x2)
        x2 = self.conv(x2)  # [batch_size, 250, seq_len-3+1, 1]
        while x2.size()[2] > 2:
            x2 = self._block(x2)
        x2 = x2.squeeze()  # [batch_size, num_filters(250)]

        # transform
        # at = x3.squeeze(1)
        # at = self.postion_embedding(at)
        # for encoder in self.encoders:
        #     at = encoder(at)
        # at = at.view(at.size(0), -1)

        # textcnn maxpoll/minpoll
        # tmax = torch.cat([self.conv_and_pool_max(x3, conv) for conv in self.convs_add], 1)
        # tmin = torch.cat([self.conv_and_pool_min(x3, conv) for conv in self.convs_add], 1)

        # 连接起来
        # x = torch.cat((x1, x2, tmax, tmin), 1)  # 连接 [batch_size, num_filters(500)]
        x = torch.cat((x1, x2), 1)  # 连接 [batch_size, num_filters(500)]
        x1 = self.fc1(x)  # [128,500+config.pad_size * config.dim_model]  # gather_label
        x2 = self.fc2(x1)

        return x2, x1
        """

        """
        # 主transform 副cnn
        x = x[0]
        x = self.embedding(x)
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embed, 1]
        x1, x2, x3 = x.split(128, dim=2)  # x1要素名称，x2要素内容, x3重要特征
        # feature
        # x3 = self.conv_region(x3)  # [batch_size, 250, seq_len-3+1, 1]
        # x3 = self.padding1(x3)  # [batch_size, 250, seq_len, 1]
        # x3 = self.relu(x3)
        # x3 = self.conv(x3)  # [batch_size, 250, seq_len-3+1, 1]
        # x3 = self.padding1(x3)  # [batch_size, 250, seq_len, 1]
        # x3 = self.relu(x3)
        # x3 = self.conv(x3)  # [batch_size, 250, seq_len-3+1, 1]
        # while x3.size()[2] > 2:
        #     x3 = self._block(x3)
        # x3 = x3.squeeze()  # [batch_size, num_filters(250)]

        # attention
        x1 = x1.squeeze(1)
        x1 = self.postion_embedding(x1)
        for encoder in self.encoders:
            x1 = encoder(x1)
        x1 = x1.view(x1.size(0), -1)

        x2 = x2.squeeze(1)
        x2 = self.postion_embedding(x2)
        for encoder in self.encoders:
            x2 = encoder(x2)
        x2 = x2.view(x2.size(0), -1)

        # 连接起来
        x = torch.cat((x1, x2), 1)  # 连接 [batch_size, num_filters(500)]
        x1 = self.fc1(x)  # [128,250+config.pad_size * config.dim_model*2]  # gather_label
        x2 = self.fc2(x1)

        return x2, x1
        """

        '''
        # dpcnn
        # x = x[0]
        # x = self.embedding(x)
        # x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embed, 1]
        x = x[0]
        x = self.embedding(x)
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embed, 1]
        x1, x2_1, x2_2, x2_3, x3 = x.split(128, dim=2)  # x1要素名称，2_1 2_2 2_3三种卷积核, x3feature
        x = x2_1
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]

        x1 = self.fc1(x)  # [128,num_filters(250)]  # gather_label
        x2 = self.fc2(x1)
        return x2, x1
        '''

        # """
        # 结构化卷积
        x = x[0]
        x = self.embedding(x)
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embed, 1]
        x1, x2_1, x2_2, x2_3, x3 = x.split(128, dim=2)  # x1要素名称，2_1 2_2 2_3三种卷积核, x3feature

        # sspcnn
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
        x2 = torch.cat((x2_1, x2_2, x2_3), 1)  # [batch_size, 3*num_filters(750)]

        # add_feature textcnn
        tmax = torch.cat([self.conv_and_pool_max(x3, conv) for conv in self.convs_add], 1)
        tmin = torch.cat([self.conv_and_pool_min(x3, conv) for conv in self.convs_add], 1)


        # add_feature transform
        # at = x1.squeeze(1)
        # at = self.postion_embedding(at)
        # for encoder in self.encoders:
        #     at = encoder(at)
        # at = at.view(at.size(0), -1)
        # 
        # ae = x3.squeeze(1)
        # ae = self.postion_embedding(ae)
        # for encoder in self.encoders:
        #     ae = encoder(ae)
        # ae = ae.view(ae.size(0), -1)

        # 连接起来
        x = torch.cat((x1, x2, tmax, tmin), 1)  # 连接 [batch_size, num_filters(1000)]
        # x = torch.cat((at, ae), 1)  # 连接 [batch_size, num_filters(1000)]
        x1 = self.fc1(x)  # [128,1000]  # gather_label
        x2 = self.fc2(x1)

        return x2, x1
        # """

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

    def conv_and_pool_max(self, x, conv):  # textcnn
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def conv_and_pool_min(self, x, conv):  # textcnn
        x = F.relu(conv(x)).squeeze(3)
        x = -F.max_pool1d(-x, x.size(2)).squeeze(2)
        return x


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''

    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
