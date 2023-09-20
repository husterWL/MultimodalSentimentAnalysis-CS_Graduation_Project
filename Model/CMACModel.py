import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50

class TextModel(nn.Module):

    def __init__(self, config):
        super(TextModel, self).__init__()

        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.trans = nn.Sequential(
            nn.Dropout(config.bert_dropout),
            nn.Linear(self.bert.config.hidden_size, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        ) 
        
        # 是否进行fine-tune
        for param in self.bert.parameters():
            if config.fixed_text_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # self.bert.init_weights()

    def forward(self, bert_inputs, masks, token_type_ids=None):
        bert_out = self.bert(input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)
        '''
        通常情况下，hidden_state 是一个包含了输入文本序列各个位置（或标记）的编码信息的张量。
        这个张量的形状通常是 [batch_size, sequence_length, hidden_size]
        '''
        hidden_state = bert_out['last_hidden_state']    #最后一个隐藏层输出
        '''
        pooler_output 是通过对[CLS]标记（通常是输入序列的第一个标记）的隐藏状态进行池化操作而得到的。
        [CLS]标记的隐藏状态会经过一个池化操作（通常是均值池化或最大池化）以产生一个固定长度的向量，该向量被认为是整个输入序列的表示。
        这个池化操作的目的是将整个序列的信息压缩成一个固定大小的向量，通常用于下游分类任务。
        '''
        pooler_out = bert_out['pooler_output']
        
        return self.trans(hidden_state), self.trans(pooler_out)


class ImageModel(nn.Module):

    def __init__(self, config):
        super(ImageModel, self).__init__()
        self.full_resnet = resnet50(pretrained=True)
        '''
        这段代码是对self.full_resnet进行修改，将其除了最后两层（全连接与分类器）所有层拼接成一个序列，并将其赋值给self.resnet_h。
        即只包含特征提取
        这里使用了*符号和list()函数，将self.full_resnet.children()（即self.full_resnet中的所有层）转换为一个列表
        并将其前4个元素拼接成一个序列。
        '''
        self.resnet_h = nn.Sequential(
            *(list(self.full_resnet.children())[:-2]),
        )

        self.resnet_p = nn.Sequential(
            list(self.full_resnet.children())[-2],
            nn.Flatten()
        )

        # 本层为特殊层，目的是为了得到较少的特征响应图(原来的2048有些过大)：
        # (batch, 2048, 7, 7) -> (batch, img_hidden_seq, middle_hidden_size)
        self.hidden_trans = nn.Sequential(
            nn.Conv2d(self.full_resnet.fc.in_features, config.img_hidden_seq, 1),
            nn.Flatten(start_dim=2),
            nn.Dropout(config.resnet_dropout),
            nn.Linear(7 * 7, config.middle_hidden_size),    # 这里的7*7是根据resnet50，原img大小为224*224的情况来的
            nn.ReLU(inplace=True)
        )

        self.trans = nn.Sequential(
            nn.Dropout(config.resnet_dropout),
            nn.Linear(self.full_resnet.fc.in_features, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )
        
        # 是否进行fine-tune
        for param in self.full_resnet.parameters():
            if config.fixed_image_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True
    '''
    forward提取全局特征与局部特征
    全局特征是最后一个卷积层的特征图
    '''
    def forward(self, imgs):
        hidden_state = self.resnet_h(imgs)  #全局特征
        feature = self.resnet_p(hidden_state)   #上面的全局特征经过倒数第二层的全连接层得到的特征

        return self.hidden_trans(hidden_state), self.trans(feature)


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # text
        self.text_model = TextModel(config)
        # image
        self.img_model = ImageModel(config)
        # attention
        self.text_img_attention = nn.MultiheadAttention(
            embed_dim=config.middle_hidden_size,
            num_heads=config.attention_nhead, 
            dropout=config.attention_dropout,
        )
        self.img_text_attention = nn.MultiheadAttention(
            embed_dim=config.middle_hidden_size,
            num_heads=config.attention_nhead, 
            dropout=config.attention_dropout
        )

        # 全连接分类器
        self.text_classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels),
            nn.Softmax(dim=1)
        )
        self.img_classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels),
            nn.Softmax(dim=1)
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, texts, texts_mask, imgs, labels=None):
        text_hidden_state, text_feature = self.text_model(texts, texts_mask)

        img_hidden_state, img_feature = self.img_model(imgs)

        '''
        这两行代码分别用于交换 PyTorch 张量中的维度顺序。
        permute 方法允许你重新排列张量的维度，以适应特定的计算或操作需求
        例如：
        text_hidden_state.permute(1, 0, 2)：
            text_hidden_state 是一个张量，假设它的形状是 [sequence_length, batch_size, hidden_size]。
            permute(1, 0, 2) 将维度重新排列为 [batch_size, sequence_length, hidden_size]。
            这个操作通常用于将序列长度（sequence_length）维度移到批处理（batch_size）维度之前，这在很多情况下是需要的，特别是在循环神经网络 (RNN) 或注意力机制中。
        '''
        text_hidden_state = text_hidden_state.permute(1, 0, 2)
        img_hidden_state = img_hidden_state.permute(1, 0, 2)

        text_img_attention_out, _ = self.img_text_attention(img_hidden_state, \
            text_hidden_state, text_hidden_state)
        text_img_attention_out = torch.mean(text_img_attention_out, dim=0).squeeze(0)
        img_text_attention_out, _ = self.text_img_attention(text_hidden_state, \
            img_hidden_state, img_hidden_state)
        img_text_attention_out = torch.mean(img_text_attention_out, dim=0).squeeze(0)

        text_prob_vec = self.text_classifier(torch.cat([text_feature, img_text_attention_out], dim=1))
        img_prob_vec = self.img_classifier(torch.cat([img_feature, text_img_attention_out], dim=1))

        prob_vec = torch.softmax((text_prob_vec + img_prob_vec), dim=1)
        pred_labels = torch.argmax(prob_vec, dim=1)

        if labels is not None:
            loss = self.loss_func(prob_vec, labels)
            return pred_labels, loss
        else:
            return pred_labels