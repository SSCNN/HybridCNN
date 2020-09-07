# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network, load_embeding, test
from importlib import import_module
import argparse
import os
from utils import build_dataset, build_iterator, get_time_dif
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--models', type=str, default='DPCNN_test',
                    help='choose a models: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='random', type=str, help='random or pre_trained')
parser.add_argument('--word', default=True, type=bool, help='True for word, False for char')
parser.add_argument('--use-premodel', default=False, type=bool, help='enable use pre-models')
parser.add_argument('--start', default='test', type=str, help='train, test, pre')
args = parser.parse_args()

if __name__ == '__main__':
    # dataset = 'bertplus_data'
    dataset = '84-85-90'

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.models  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding, args.use_premodel)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    # class_map
    # class_map = dict()
    # class_map_list = [x.strip().split('\t') for x in open(config.class_map_path, encoding='utf-8').readlines()]
    # for group in class_map_list:
    #     small_index = config.class_list.index(group[0])
    #     big_index = config.gather_class_list.index(group[1])
    #     class_map[small_index] = big_index

    # loading feature_map
    feature_map = dict()
    with open(config.feature_map_path, encoding='utf-8')as f:  # 特征重要度排名名单
        data = f.readlines()
        for i, line in enumerate(data):
            if i % 2 == 0:
                gather_label = line.strip()
                feature_map[gather_label] = []
            else:
                feature = line.strip().split('\t')
                for _ in feature:
                    feature_map[gather_label].append(int(_[0]))
    print('loaded feature map')

    vocab, train_data, dev_data, test_data = build_dataset(config, args.word, feature_map)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)

    if args.use_premodel:  # 使用预训练模型

        pre_model = x.Model(config).to(config.device)
        pretrained_dict = torch.load(config.load_path)
        # 1. filter out unnecessary keys
        premodel_dict = {k: v for k, v in pretrained_dict.items() if k != 'fc.weight'}
        premodel_dict = {k: v for k, v in premodel_dict.items() if k != 'fc.bias'}
        # 2. overwrite entries in the existing state dict
        pre_model.load_state_dict(premodel_dict, strict=False)  # 加载模型权重，strick=False表示不加载无效层
        print(pre_model.parameters)

        if args.start == 'train':
            train(config, pre_model, train_iter, dev_iter, test_iter)
        elif args.start == 'test':
            test(config, pre_model, test_iter)
        os._exit(1)

    model = x.Model(config).to(config.device)

    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    if args.start == 'train':
        train(config, model, train_iter, dev_iter, test_iter)
    elif args.start == 'test':
        test(config, model, test_iter)

