import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CE_weighted(nn.Module):
    """
    手动计算交叉熵和KL散度损失并按照style_label加权
    """
    def __init__(self):
        super(CE_weighted, self).__init__()
        self.logsoftmax_func = nn.LogSoftmax(dim=2)
        self.nll_func = nn.NLLLoss()

    def forward(self, logit, cap, cap_len, weight):
        target = cap[:, 1:]  # (batch_size, 21)
        cap_len = cap_len - 1

        logit = logit  # (batch_size, 21, vocab_size)
        logit_logsoftmax = self.logsoftmax_func(logit)  # (batch_size, 21, vocab_size)
        # weight = torch.ones(logit.size(0)).to(device)
        weight_ce = weight.unsqueeze(1).expand(logit.size(0), logit.size(1)).unsqueeze(2).expand(logit.size())  # (batch_size, 21, vocab_size)
        weighted_logit_logsoftmax = logit_logsoftmax*weight_ce
        target_cal = pack_padded_sequence(target, cap_len, batch_first=True, enforce_sorted=False)[0]
        logit_cal = pack_padded_sequence(weighted_logit_logsoftmax, cap_len, batch_first=True, enforce_sorted=False)[0]
        loss_ce = self.nll_func(logit_cal, target_cal)

        return loss_ce

class CE_KL(nn.Module):
    """
    compute the crossentropy loss and KL loss
    """
    def __init__(self):
        super(CE_KL, self).__init__()
        self.ce = nn.CrossEntropyLoss().to(device)

    def forward(self, logit, mu, sigma2, cap, cap_len):
        target = cap[:, 1:]
        cap_len = cap_len - 1

        target = pack_padded_sequence(target, cap_len, batch_first=True, enforce_sorted=False)[0]
        logit = pack_padded_sequence(logit, cap_len, batch_first=True, enforce_sorted=False)[0]

        # reconstruct loss
        loss_ce = self.ce(logit, target)

        # KL-divergence (entire batch)
        loss_kl = ((-0.5 * torch.sum(1 + sigma2 - torch.exp(sigma2) - mu**2))/mu.size(0))

        return loss_ce, loss_kl  # 目标是最小化1.重建损失 2.后验和标准正态的距离 即可，不用纠结具体项的正负


class CE_KL_weighted(nn.Module):
    """
    手动计算交叉熵和KL散度损失并使其可以按照style_label加权
    """
    def __init__(self):
        super(CE_KL_weighted, self).__init__()
        self.logsoftmax_func = nn.LogSoftmax(dim=2)
        self.nll_func = nn.NLLLoss()

    def forward(self, logit, mu, sigma2, cap, cap_len, weight):
        target = cap[:, 1:]  # (batch_size, 21)
        cap_len = cap_len - 1

        logit = logit  # (batch_size, 21, vocab_size)
        logit_logsoftmax = self.logsoftmax_func(logit)  # (batch_size, 21, vocab_size)
        # weight = torch.ones(logit.size(0)).to(device)
        weight_ce = weight.unsqueeze(1).expand(logit.size(0), logit.size(1)).unsqueeze(2).expand(logit.size())  # (batch_size, 21, vocab_size)
        weighted_logit_logsoftmax = logit_logsoftmax*weight_ce
        target_cal = pack_padded_sequence(target, cap_len, batch_first=True, enforce_sorted=False)[0]
        logit_cal = pack_padded_sequence(weighted_logit_logsoftmax, cap_len, batch_first=True, enforce_sorted=False)[0]
        loss_ce = self.nll_func(logit_cal, target_cal)

        # weight = torch.ones(mu.size(0)).to(device)
        # weight_kl = weight.unsqueeze(1).expand(mu.size())
        # loss_kl = ((-0.5 * torch.sum((1 + sigma2 - torch.exp(sigma2) - mu ** 2)*weight_kl)) / mu.size(0))
        loss_kl = ((-0.5 * torch.sum((1 + sigma2 - torch.exp(sigma2) - mu ** 2))) / mu.size(0))
        return loss_ce, loss_kl


class CE_KL_weighted_1(nn.Module):
    """
    手动计算交叉熵和KL散度损失并使其可以按照style_label加权
    """
    def __init__(self):
        super(CE_KL_weighted_1, self).__init__()
        self.logsoftmax_func = nn.LogSoftmax(dim=2)
        self.nll_func = nn.NLLLoss()

    def forward(self, logit, mu, sigma2, mu_pri, sigma2_pri, cap, cap_len, weight):
        target = cap[:, 1:]  # (batch_size, 21)
        cap_len = cap_len - 1

        logit = logit  # (batch_size, 21, vocab_size)
        logit_logsoftmax = self.logsoftmax_func(logit)  # (batch_size, 21, vocab_size)
        # weight = torch.ones(logit.size(0)).to(device)
        weight_ce = weight.unsqueeze(1).expand(logit.size(0), logit.size(1)).unsqueeze(2).expand(logit.size())  # (batch_size, 21, vocab_size)
        weighted_logit_logsoftmax = logit_logsoftmax*weight_ce
        target_cal = pack_padded_sequence(target, cap_len, batch_first=True, enforce_sorted=False)[0]
        logit_cal = pack_padded_sequence(weighted_logit_logsoftmax, cap_len, batch_first=True, enforce_sorted=False)[0]
        loss_ce = self.nll_func(logit_cal, target_cal)

        # weight = torch.ones(mu.size(0)).to(device)
        # weight_kl = weight.unsqueeze(1).expand(mu.size())
        # loss_kl = ((-0.5 * torch.sum((1 + sigma2 - torch.exp(sigma2) - mu ** 2)*weight_kl)) / mu.size(0))
        loss_kl = (-0.5 * torch.sum(1 + sigma2 - sigma2_pri - (torch.exp(sigma2)/torch.exp(sigma2_pri)) - (((mu-mu_pri)**2)/torch.exp(sigma2_pri))))/mu.size(0)
        return loss_ce, loss_kl


class CE_selected(nn.Module):
    def __init__(self):
        super(CE_selected, self).__init__()
        self.ce = nn.CrossEntropyLoss().to(device)

    def forward(self, logit, cap, cap_len, style_label):

        style_dims = torch.Tensor([i for i in range(style_label.size(0)) if int(style_label[i]) != 4]).long().to(device)
        if int(style_dims.size(0)) == 0:
            return torch.Tensor([0])
        logit = torch.index_select(logit, 0, style_dims)
        cap = torch.index_select(cap, 0, style_dims)
        cap_len = torch.index_select(cap_len, 0, style_dims)

        target = cap[:, 1:]
        cap_len = cap_len - 1

        target = pack_padded_sequence(target, cap_len, batch_first=True, enforce_sorted=False)[0]
        logit = pack_padded_sequence(logit, cap_len, batch_first=True, enforce_sorted=False)[0]

        # reconstruct loss
        loss_ce = self.ce(logit, target)
        return loss_ce


class KL_selected(nn.Module):
    def __init__(self):
        super(KL_selected, self).__init__()

    def forward(self, mu, sigma2, style_label):
        style_dims = torch.Tensor([i for i in range(style_label.size(0)) if int(style_label[i]) != 4]).long().to(device)
        if int(style_dims.size(0)) == 0:
            return torch.Tensor([0])
        mu = torch.index_select(mu, 0, style_dims)
        sigma2 = torch.index_select(sigma2, 0, style_dims)
        loss_kl_style = ((-0.5 * torch.sum(1 + sigma2 - torch.exp(sigma2) - mu ** 2)) / mu.size(0))
        return loss_kl_style

class KL_selected_1(nn.Module):
    def __init__(self):
        super(KL_selected_1, self).__init__()

    def forward(self, mu, sigma2, mu_pri, sigma2_pri, style_label):
        style_dims = torch.Tensor([i for i in range(style_label.size(0)) if int(style_label[i]) != 4]).long().to(device)
        if int(style_dims.size(0)) == 0:
            return torch.Tensor([0])
        mu = torch.index_select(mu, 0, style_dims)
        sigma2 = torch.index_select(sigma2, 0, style_dims)
        mu_pri = torch.index_select(mu_pri, 0, style_dims)
        sigma2_pri = torch.index_select(sigma2_pri, 0, style_dims)
        loss_kl_style = (-0.5 * torch.sum(1 + sigma2 - sigma2_pri - (torch.exp(sigma2)/torch.exp(sigma2_pri)) - (((mu-mu_pri)**2)/torch.exp(sigma2_pri))))/mu.size(0)
        return loss_kl_style

class Style_loss_selected(nn.Module):
    def __init__(self):
        super(Style_loss_selected, self).__init__()
        self.ce = nn.CrossEntropyLoss().to(device)

    def forward(self, style_pred, style_label):
        style_dims = torch.Tensor([i for i in range(style_label.size(0)) if int(style_label[i]) != 4]).long().to(device)
        if int(style_dims.size(0)) == 0:
            return torch.Tensor([0])
        style_pred = torch.index_select(style_pred, 0, style_dims)
        style_label = torch.index_select(style_label, 0, style_dims)
        return self.ce(style_pred, style_label)


class Style_loss(nn.Module):

    def __init__(self):
        super(Style_loss, self).__init__()
        self.ce = nn.CrossEntropyLoss().to(device)

    def forward(self, style_pred, style_label):
        return self.ce(style_pred, style_label)


class Style_loss_weighted(nn.Module):

    def __init__(self):
        super(Style_loss_weighted, self).__init__()
        self.logsoftmax_func = nn.LogSoftmax(dim=1)
        self.nll_func = nn.NLLLoss()

    def forward(self, style_pred, style_label, weight):
        pred_logsoftmax = self.logsoftmax_func(style_pred)
        weight_sc = weight.unsqueeze(1).expand(style_pred.size())
        weighted_pred_logsoftmax = pred_logsoftmax*weight_sc
        return self.nll_func(weighted_pred_logsoftmax, style_label)
