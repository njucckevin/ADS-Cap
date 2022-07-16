import torch
import torch.nn as nn
import torch.nn.functional as F
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Contrastive_loss(nn.Module):

    def __init__(self, config):
        super(Contrastive_loss, self).__init__()
        self.ce = nn.CrossEntropyLoss().to(device)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8).to(device)
        self.temperature = config.temperature

    def forward(self, obj_vec, res_vec, style_label, feat_vec):
        """
        :param obj_vec: (batch_size, align_dim)
        :param res_vec: (batch_size, align_dim)
        :param style_label: (batch_size)
        :param feat_vec: (batch_size, align_dim)
        """
        samples_style = torch.Tensor([1 if int(item) != 4 else 0 for item in style_label]).to(device)

        # 计算正例间的距离，表示为 (batch_size, 1)
        pos_dist = self.cos(obj_vec, res_vec)  # (batch_size)
        pos_dist[[i for i in range(style_label.size(0)) if int(samples_style[i])==1]] = 1  # 风格样例没有resnet特征，因此不计算相似度
        loss_cos = (1-pos_dist).mean()
        pos_dist = pos_dist.unsqueeze(1)  # (batch_size, 1)

        # 根据feat_vec计算负例间的距离，表示为 (batch_size, batch_size),负例可以是res-res、obj-obj或res-obj
        feat_vec_norm = F.normalize(feat_vec, p=2, dim=1)
        neg_dist = torch.mm(feat_vec_norm, feat_vec_norm.t())  # (batch_size, batch_size) 表示第i个特征和第j个特征的相似度（根据feat_vec，特征随机用res或obj表示）

        # 对neg_list矩阵进行修正


        # 通过交叉熵计算对比学习的损失
        logits = torch.cat([pos_dist, neg_dist], dim=1)  # (batch_size, batch_size+1)
        logits = logits / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        return self.ce(logits, labels), loss_cos

    """
    def forward(self, obj_vec, res_vec, style_label):

        samples_style = torch.Tensor([1 if int(item) != 4 else 0 for item in style_label]).to(device)

        # 计算正例间的距离，表示为 (batch_size, 1)
        pos_dist = self.cos(obj_vec, res_vec)  # (batch_size)
        pos_dist[[i for i in range(style_label.size(0)) if int(samples_style[i])==1]] = 1  # 风格样例没有resnet特征，因此不计算相似度
        loss_cos = (1-pos_dist).mean()
        pos_dist = pos_dist.unsqueeze(1)  # (batch_size, 1)

        # 计算负例间的距离，表示为 (batch_size, batch_size),负例可以是res-res、obj-obj或res-obj
        random_weight_obj = torch.Tensor([1 if samples_style[i]==1 else random.randint(0, 1) for i in range(obj_vec.size(0))]).to(device)
        random_weight_res = 1 - random_weight_obj
        neg_vec_1 = obj_vec*(random_weight_obj.unsqueeze(1).expand(obj_vec.shape))+res_vec*(random_weight_res.unsqueeze(1).expand(res_vec.shape))
        neg_vec_1_norm = F.normalize(neg_vec_1, p=2, dim=1)  # (batch_size, align_dim)
        random_weight_obj = torch.Tensor([1 if samples_style[i]==1 else random.randint(0, 1) for i in range(obj_vec.size(0))]).to(device)
        random_weight_res = 1 - random_weight_obj
        neg_vec_2 = obj_vec*(random_weight_obj.unsqueeze(1).expand(obj_vec.shape))+res_vec*(random_weight_res.unsqueeze(1).expand(res_vec.shape))
        neg_vec_2_norm = F.normalize(neg_vec_2, p=2, dim=1)  # (batch_size, align_dim)
        neg_dist = torch.mm(neg_vec_1_norm, neg_vec_2_norm.t())  # (batch_size, batch_size) 表示第i个特征和第j个特征的相似度，特征随机用res或obj表示

        # 通过交叉熵计算对比学习的损失
        logits = torch.cat([pos_dist, neg_dist], dim=1)  # (batch_size, batch_size+1)
        logits = logits / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        return self.ce(logits, labels), loss_cos
    """