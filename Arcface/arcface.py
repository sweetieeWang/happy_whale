from pickle import TRUE
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet 

class Effnet(nn.Module):
    def __init__(self, num_classes = 15587, pretrained = True):
        super(Effnet, self).__init__()
        if pretrained:
            self.efficentnet = EfficientNet.from_pretrained('efficientnet-b5')
        else:
            self.efficentnet = EfficientNet.from_name('efficientnet-b5')
        self.feature = self.efficentnet._fc.in_features
        self.efficentnet._fc = nn.Linear(in_features=self.feature,out_features=512,bias=True)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, 512))
        # self.cos = nn.Linear(in_features = 512, out_features=num_classes)
        nn.init.xavier_uniform_(self.weight)

    def forward(self,x):
        features = self.efficentnet(x)
        logits = F.linear(F.normalize(features), F.normalize(self.weight))
        return logits

class ArcFace(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        # in_features: int,
        num_classes: int = 1000,
        s: float = 30.0,
        m: float = 0.5,
        easy_margin: bool = False,
        ls_eps: float = 0.0,
    ):
        super(ArcFace, self).__init__()
        #self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # 由 cosθ 计算相应的 sinθ
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # 展开计算 cos(θ+m) = cosθ*cosm - sinθ*sinm, 其中包含了 Target Logit (cos(θyi+ m)) (由于输入特征 xi 的非真实类也参与了计算, 最后计算新 Logit 时需使用 One-Hot 区别)
        phi = cosine * self.cos_m - sine * self.sin_m  
        # 是否松弛约束??
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # 将 labels 转换为独热编码, 用于区分是否为输入特征 xi 对应的真实类别 yi
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # 计算新 Logit
        #  - 只有输入特征 xi 对应的真实类别 yi (one_hot=1) 采用新 Target Logit cos(θ_yi + m)
        #  - 其余并不对应输入特征 xi 的真实类别的类 (one_hot=0) 则仍保持原 Logit cosθ_j
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # can use torch.where if torch.__version__  > 0.4
        # 使用 s rescale 放缩新 Logit, 以馈入传统 Softmax Loss 计算
        output *= self.s
 
        return output


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
    def __init__(self, class_num=1000, alpha=None, gamma=2, size_average=True):
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
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss