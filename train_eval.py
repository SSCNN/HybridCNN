# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter
# from focal_loss import FocalLoss
from another_focal_loss import FocalLoss
from dice_loss import DiceLoss


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if name == 'encoder.attention.fc_Q.weight':  # 不初始化transform
            break
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = 1  # 记录允许的早停次数，默认是1

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')  # 验证集最佳loss
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    focalLoss = FocalLoss(config.num_classes)  # focal loss
    # diceloss = DiceLoss(config.num_classes)  # dice loss
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels, gather_labels) in enumerate(train_iter):
            outputs, gather_outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)  # 交叉熵
            # loss = focalLoss(outputs, labels)  # focal loss
            # loss = diceloss(outputs, labels)  # dice loss
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，并且已经早停过一次了，结束训练
                if scheduler == 0:
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
                else:
                    print('change lr_rate = {} and require_improvement = {}'.format(config.learning_rate / 10, 3000))
                    scheduler -= 1  # 减一次早停回合
                    config.require_improvement = 3000  # 更改早停回合数
                    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate / 10)  # 重新定义学习率
        if flag:
            break
    writer.close()
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    # test_acc, test_loss, test_report, test_confusion, wrong_list, wrong_number, wrong_rate1 = evaluate(config, models, test_iter, test=True)
    test_acc, test_loss, wrong_list, wrong_number, wrong_rate1, test_report = evaluate(config, model, test_iter,
                                                                                       test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    # print("Confusion Matrix...")
    # print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    return str(test_loss.cpu().item()), str(test_acc)
    print('----------------------------------')
    print('pre, lab, pre_galab, galab')
    wrong_dict = {}
    for line in wrong_list:
        if line[1] not in wrong_dict.keys():
            wrong_dict[line[1]] = 1
        else:
            wrong_dict[line[1]] += 1
        print(line)
    for key, value in wrong_dict.items():
        print(key + ' ' + str(value) + '条')
    print('test number 180300')
    print('小税号预测错误条数 {}'.format(len(wrong_list)))
    print('在预测错误的税号中，小税号不在正确大税号下的概率{}%'.format(wrong_rate1 * 100))


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)  # 预测小税号
    gather_predict_all = np.array([], dtype=int)  # 预测大税号
    labels_all = np.array([], dtype=int)
    gather_labels_all = np.array([], dtype=int)
    focalLoss = FocalLoss(config.num_classes)
    diceloss = DiceLoss(config.num_classes)  # dice loss
    with torch.no_grad():
        for texts, labels, gather_labels in data_iter:
            outputs, gather_outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            # loss = focalLoss(outputs, labels)  # dic loss
            # loss = diceloss(outputs, labels)  # dice loss
            loss_total += loss

            labels = labels.data.cpu().numpy()  # label
            gather_labels = gather_labels.data.cpu().numpy()  # 大分类税号

            predic = torch.max(outputs.data, 1)[1].cpu().numpy()  # predic
            gather_predic = torch.max(gather_outputs.data, 1)[1].cpu().numpy()  # predic_gather

            labels_all = np.append(labels_all, labels)
            gather_labels_all = np.append(gather_labels_all, gather_labels)
            predict_all = np.append(predict_all, predic)
            gather_predict_all = np.append(gather_predict_all, gather_predic)

    haiguan_labels_all = [config.class_list[x] for x in labels_all]
    haiguan_predic_all = [config.class_list[x] for x in predict_all]
    haiguan_gather_labels_all = [config.gather_class_list[x] for x in gather_labels_all]
    haiguan_gather_predict_labels_all = [config.gather_class_list[x] for x in gather_predict_all]
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        # report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        report = metrics.classification_report(labels_all, predict_all, digits=4)
        # confusion = metrics.confusion_matrix(labels_all, predict_all)
        # class_map = {}  # 大小税号映射表
        wrong_number = 0  # 小税号预测错误条数
        # '''
        # class_map_list = [x.strip().split('\t') for x in open(config.class_map_path, encoding='utf-8').readlines()]
        # for lin in class_map_list:
        #     class_map[lin[0]] = lin[1]
        list1, list2, list3, list4, index_list = [], [], [], [], []  # 装结果的5列表
        for i, (pre, lab, galab, pre_galab) in enumerate(
                zip(haiguan_predic_all, haiguan_labels_all, haiguan_gather_labels_all,
                    haiguan_gather_predict_labels_all)):
            if pre_galab == galab and pre == lab:  # 全对
                list1.append((pre, lab, pre_galab, galab))
            elif pre_galab == galab and pre != lab:  # 大对小不对
                index_list.append([str(i), pre, lab, galab])
                list2.append((pre, lab, pre_galab, galab))
            elif pre_galab != galab and pre == lab:  # 大不对小对
                list3.append((pre, lab, pre_galab, galab))
            elif pre_galab != galab and pre != lab:  # 大不对小不对
                index_list.append([str(i), pre, lab, galab])
                list4.append((pre, lab, pre_galab, galab))
        print('全对有{}条'.format(len(list1)))
        print('大对小不对有{}条'.format(len(list2)))
        print('大不对小对有{}条'.format(len(list3)))
        print('大不对小不对有{}条'.format(len(list4)))
        """
        with open('./84-85-90/base_wrong.txt', 'w', encoding='utf-8')as f:
            f.write('index, prelab, lab, galab')
            f.write('\n')
            for line in index_list:
                line = '\t'.join(line)
                f.write(line)
                f.write('\n')
        print('index_finish')
        """
        wrong_list = list2 + list4  # 小税号预测错误汇总
        test_number = 0
        for line in wrong_list:
            if line[3] not in line[0]:  # 预测的小税号不在正确的大税号下
                test_number += 1
        print('预测小税号错误中，小税号不在正确大税号属性下有{}条'.format(test_number))
        print('\n')
        print('\n')
        print('\n')
        # '''
        wrong_rate1 = test_number / len(wrong_list)
        # return acc, loss_total / len(data_iter), report, confusion, wrong_list, wrong_number, wrong_rate1
        return acc, loss_total / len(data_iter), wrong_list, wrong_number, wrong_rate1, report
    return acc, loss_total / len(data_iter)


def load_embeding(model, train_iter):
    model.eval()
    loss_total = 0
    text_all = np.array([], dtype=float)
    with torch.no_grad():
        for texts, labels, gather_labels in train_iter:
            outputs = model(texts).cpu()
            labels = labels.data.cpu().numpy()
            gather_label = gather_labels.data.cpu().numpy()
            for output, label, gather_label in zip(outputs, labels, gather_labels):
                text_all = np.append(text_all, (output, label))
    return text_all
