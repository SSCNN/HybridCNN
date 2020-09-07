# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import jieba
import re
from random import random

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            name = lin.split('\t')[4]
            text = eval(lin.split('\t')[5])
            text1 = [key for key in text.keys()]
            text2 = [value for value in text.values()]
            content = name + '|' + '|'.join(text1 + text2)
            content = ' '.join(jieba.lcut(content))
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word, feature_map):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
        print(vocab)
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, feature_map, class_list, gather_class_list, pad_size=32, feature_number=2):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                # 正常操作
                # name = lin.split('\t')[4]
                # text = eval(lin.split('\t')[5])
                # text = [value for value in text.values()]
                # content = name + '|' + '|'.join(text)
                # content = ' '.join(jieba.lcut(content))

                # bertplus_data要素内容和要素名称分开读取
                name = lin.split('\t')[4]
                text = eval(lin.split('\t')[5])
                content1 = ['商品名称'] + [key for key in text.keys()]  # 要素名称
                content2 = [name] + [value for value in text.values()]  # 要素内容

                gather_label = lin.split('\t')[3]
                if gather_label in feature_map.keys():
                    feature_id = feature_map[gather_label]
                    feature_name = []
                    feature_text = []
                    try:
                        # 按照feature_map选前两位
                        feature_name += [content1[_] for _ in feature_id[:feature_number]]
                        feature_text += [content2[_] for _ in feature_id[:feature_number]]
                    except:
                        # print(content1)
                        # print(content2)
                        # print(gather_label)
                        feature_name = [content1[0]] + [content1[3]]
                        feature_text = [content2[0]] + [content2[3]]
                else:
                    try:
                        feature_name = [content1[0]] + [content1[3]]
                        feature_text = [content2[0]] + [content2[3]]
                    except:
                        feature_name = [content1[0]]
                        feature_text = [content2[0]]

                feature_content = ''
                for name, text in zip(feature_name, feature_text):
                    feature_content += name + text + '|'
                feature_content = feature_content[:-1]
                feature_content = ' '.join(jieba.lcut(feature_content))

                content1 = '|'.join(content1)

                content2_1 = '|'.join(content2)
                content2_2 = '||'.join(content2)
                content2_3 = '|||'.join(content2)
                content2 = '|'.join(content2)

                content1 = jieba.lcut(content1)

                content2 = jieba.lcut(content2)
                content2_1 = jieba.lcut(content2_1)
                content2_2 = jieba.lcut(content2_2)
                content2_3 = jieba.lcut(content2_3)

                index_list = []
                for index, word in enumerate(content2):  # 定位要素信息'|'的位置，获得要素分割地址
                    if word == '|':
                        index_list.append(index)

                content1 = ' '.join(content1)

                content2 = ' '.join(content2)
                content2_1 = ' '.join(content2_1)
                content2_2 = ' '.join(content2_2)
                content2_3 = ' '.join(content2_3)

                content2_1 = re.sub('\|', PAD, content2_1)  # 不做pading
                content2_2 = re.sub('\|', PAD, content2_2)
                content2_3 = re.sub('\|', PAD, content2_3)

                # bertplus_data只加载要素名称
                # content = eval(content)
                # content = [keys for keys in content.keys()]
                # content = '商品名称|'+'|'.join(content)

                # 正常
                label = lin.split('\t')[2]
                if label in class_list:
                    label = class_list.index(label)
                else:
                    print('sth wrong')
                    print(line)
                    continue

                # 大税号分类
                # label = lin.split('\t')[3]
                # label = class_list.index(label)

                gather_label = lin.split('\t')[3]
                if gather_label in gather_class_list:
                    gather_label = gather_class_list.index(gather_label)
                else:
                    print('sth wrong')
                    print(line)
                    continue

                words_line = []

                # 正常2id
                # token = tokenizer(content)
                # seq_len = len(token)

                """
                # 用两个token做2id
                token1 = tokenizer(content1)
                token2 = tokenizer(content2)
                seq_len1 = len(token1)
                seq_len2 = len(token2)

                # 分别对两个token进行pad操作
                for i, token in enumerate([token1, token2]):
                    if pad_size:
                        if len(token) < pad_size:
                            seq_len = len(token)
                            token.extend([PAD] * (pad_size - len(token)))
                        else:
                            token = token[:pad_size]
                            seq_len = pad_size
                        if i == 0:
                            token1 = token
                            seq_len1 = seq_len
                        elif 1 == 1:
                            token2 = token
                            seq_len2 = seq_len

                index_list.append(seq_len2)  # 加一个最后长度，防止把pad算入, 并填充至20
                if len(index_list) < 20:
                    index_list.extend([-1] * (20 - len(index_list)))
                else:
                    print('this sentense index over')
                token = token1 + token2
                """

                """
                # 正常操作：
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = len(token)
                """
                # """
                # 4组不同的pading
                token1 = tokenizer(content1)
                token2_1 = tokenizer(content2_1)
                token2_2 = tokenizer(content2_2)
                token2_3 = tokenizer(content2_3)
                token3 = tokenizer(feature_content)

                seq_len1 = len(token1)
                seq_len2_1 = len(token2_1)
                seq_len2_2 = len(token2_2)
                seq_len2_3 = len(token2_3)
                seq_len3 = len(token3)

                # 分别对4个token进行pad操作
                for i, token in enumerate([token1, token2_1, token2_2, token2_3, token3]):
                    if pad_size:
                        if len(token) < pad_size:
                            seq_len = len(token)
                            token.extend([PAD] * (pad_size - len(token)))
                        else:
                            token = token[:pad_size]
                            seq_len = pad_size
                        if i == 0:
                            token1 = token
                            seq_len1 = seq_len
                        elif i == 1:
                            token2_1 = token
                            seq_len2_1 = seq_len
                        elif i == 2:
                            token2_2 = token
                            seq_len2_2 = seq_len
                        elif i == 3:
                            token2_3 = token
                            seq_len2_3 = seq_len
                        elif i == 4:
                            token3 = token
                            seq_len3 = seq_len

                token = token1 + token2_1 + token2_2 + token2_3 + token3
                # """

                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                # contents.append((words_line, int(label), seq_len, int(gather_label)))  # 正常的2id
                contents.append((words_line, int(label), seq_len1, seq_len2_1, seq_len2_2, seq_len2_3, int(gather_label)))  # 两条语句的2id
        return contents  # [([...], 0, 32, 32，'8400'), ([...], 1), ...]

    train = load_dataset(config.train_path, feature_map, config.class_list, config.gather_class_list, config.pad_size)
    dev = load_dataset(config.dev_path, feature_map, config.class_list, config.gather_class_list, config.pad_size)
    test = load_dataset(config.test_path, feature_map, config.class_list, config.gather_class_list, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)  # 文本id
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)  # label
        z = torch.LongTensor([_[6] for _ in datas]).to(self.device)  # gather_label  3或者4或者6

        """
        # 正常操作
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y, z

        # """
        # 两层token转tensor
        seq_len1 = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        seq_len2_1 = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        seq_len2_2 = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        seq_len2_3 = torch.LongTensor([_[5] for _ in datas]).to(self.device)
        # index_list = torch.LongTensor([_[5] for _ in datas]).to(self.device)
        return (x, seq_len1, seq_len2_1, seq_len2_2, seq_len2_3), y, z
        # """

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
