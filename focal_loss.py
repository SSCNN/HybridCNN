import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# class FocalLoss(nn.Module):
#     def __init__(self, ):
#         super(FocalLoss, self).__init__()
#         self.device = torch.device("cuda:" + str(0))
#         self.focal_loss_alpha = 0.8
#         self.focal_loss_gamma = 2
#
#     def forward(self, inputs, targets):
#         gpu_targets = targets.cuda()
#         alpha_factor = torch.ones(gpu_targets.shape).cuda() * self.focal_loss_alpha
#         alpha_factor = torch.where(torch.eq(gpu_targets, 1), alpha_factor, 1. - alpha_factor)
#         focal_weight = torch.where(torch.eq(gpu_targets, 1), 1. - inputs, inputs)
#         focal_weight = alpha_factor * torch.pow(focal_weight, self.focal_loss_gamma)
#         targets = targets.type(torch.FloatTensor)
#         inputs = inputs.cuda()
#         targets = targets.cuda()
#         bce = F.binary_cross_entropy(inputs, targets)
#         focal_weight = focal_weight.cuda()
#         cls_loss = focal_weight * bce
#         return cls_loss.sum()
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
