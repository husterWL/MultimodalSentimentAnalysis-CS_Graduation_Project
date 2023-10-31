import torch
import torch.nn as nn

# 定义注意力机制
class AttentionMechanism(nn.Module):
    def __init__(self, text_feature_dim, local_feature_dim):
        super(AttentionMechanism, self).__init__()
        self.W = nn.Linear(text_feature_dim, local_feature_dim) #按维度设定线性层

    def forward(self, text_features, local_features):
        # 计算相关性分数
        scores = torch.matmul(self.W(text_features), local_features.transpose(1, 2))
        
        # 计算注意力权重
        attention_weights = torch.softmax(scores, dim=2)
        
        # 使用注意力权重融合局部特征
        weighted_local_features = torch.matmul(attention_weights, local_features)
        
        return weighted_local_features

# 定义多模态融合模块
class MultiModalFusion(nn.Module):
    def __init__(self, text_feature_dim, local_feature_dim):
        super(MultiModalFusion, self).__init__()
        self.attention = AttentionMechanism(text_feature_dim, local_feature_dim)    #按维度设定注意力

    def forward(self, text_features, local_features):
        # 使用注意力机制融合局部特征和文本特征
        fused_features = self.attention(text_features, local_features)
        
        return fused_features

# 使用示例
text_feature_dim = 256  # 替换为文本联合特征的维度
local_feature_dim = 128  # 替换为局部特征的维度

# 创建模型实例
fusion_model = MultiModalFusion(text_feature_dim, local_feature_dim)

# 输入文本联合特征和局部特征
text_features = torch.randn(1, text_feature_dim)  # 示例的文本联合特征
local_features = torch.randn(1, local_feature_dim, 10)  # 示例的局部特征，假设有10个局部特征

# 使用多模态融合模块融合特征
fused_features = fusion_model(text_features, local_features)

# 现在，fused_features 包含了被关注的视觉特征，它是局部特征与文本的相关性加权融合的结果



'''
2023年10月29日
att_weights与image_features做了逐元素相乘，生成了一个注意力权重矩阵，然后将注意力权重应用于image_features，得到融合特征。
'''
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, text_model, img_model, img_hidden_seq):
        super().__init__()
        self.text_model = text_model
        self.img_model = img_model
        self.attention = nn.Sequential(
            nn.Linear(text_model.shared_dimension, img_hidden_seq, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(img_hidden_seq, text_model.shared_dimension)
        )
    
    def forward(self, words, phrases, docs, images):
        text_features = self.text_model(words, phrases, docs)
        text_features = text_features.view(text_features.size(0), -1, 1, text_features.size(1))
        image_features = self.img_model(images).permute(0,2,3,1).contiguous().view(image_features.size(0),-1,text_features.size(1))
        att_weights = self.attention(text_features).view(text_features.size(0),-1,1,1)
        att_weights = att_weights.expand(-1,-1,image_features.size(2),-1)
        fusion_features = att_weights * image_features
        fusion_features = fusion_features.sum(dim=1)
        return fusion_features