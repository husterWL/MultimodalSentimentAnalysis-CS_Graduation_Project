import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50


class TextModel(nn.Module):

    def __init__(self, config): #接受一个config参数，用于指定预训练模型的参数
        super(TextModel, self).__init__()   #这是调用父类（nn.module）的构造函数，并将self传递给该构造函数

        #使用预训练模型的配置和权重来创建一个新的模型，并将其初始化为预训练模型的状态。
        #config.bert_name是一个字符串，指定要使用的预训练模型的名称。
        self.bert = AutoModel.from_pretrained(config.bert_name)
        '''
        AutoModel.from_pretrained()函数接受一个预训练模型的名称，并使用该名称从transformers模块中导入预训练模型。
        AutoModel.from_pretrained()函数将使用导入的预训练模型的配置和权重来创建新的模型，并将其初始化为预训练模型的状态。
        '''
        self.trans = nn.Sequential( #一个连续的容器；创建了一个前向传播函数，并将其添加到self实例对象中。
            nn.Dropout(config.bert_dropout),    #nn.Dropout()是一个nn.Module对象，用于在神经网络模型的训练过程中随机删除一部分神经元，以防止过拟合。
            nn.Linear(self.bert.config.hidden_size, config.middle_hidden_size), #nn.Linear()是一个nn.Module对象，用于将神经网络模型中的输入数据映射为输出数据。
            nn.ReLU(inplace=True)   #nn.ReLU(inplace=True)是一个nn.Module对象，用于将神经网络模型中的输入数据映射为输出数据，并在训练过程中使用 inplace 算法。
        ) 
        '''
            self.trans 是一个由几个层组成的序列，其中包括一个丢弃层（nn.Dropout）、一个线性层（nn.Linear）和一个 ReLU 激活函数。
            '''
        
        # 是否进行fine-tune
        for param in self.bert.parameters():    #循环遍历self.bert实例对象中的所有参数，并将它们设置为不需要梯度的状态。
            '''
            如果config.fixed_text_model_params为True，则self.bert实例对象中的所有参数都将被设置为不需要梯度的状态；
            否则，self.bert实例对象中的所有参数都将被设置为需要梯度的状态。
            '''
            if config.fixed_text_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        #self.bert.init_weights()

    def forward(self, bert_inputs, masks, token_type_ids=None):#是PyTorch 模型类中必须的一个方法，用于执行模型的前向传播
        #bert_inputs 和 masks 分别是输入到 BERT 模型的输入张量和掩码张量，token_type_ids 是可选的，表示不同类型的 tokens（如句子 A 和句子 B）
        '''
        assert检查语句：bert_inputs和masks的形状是否相同。如果bert_inputs和masks的形状不相同，则forward函数将抛出一个AssertionError异常，并输出错误信息。
        '''
        assert bert_inputs.shape == masks.shape, 'error! bert_inputs and masks must have same shape!'
        #使用预先定义的 BERT 模型 self.bert 来进行前向传播。
        bert_out = self.bert(input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)
        '''
        input_ids 是输入 tokens 的索引
        token_type_ids 是区分不同句子的 token 类型
        attention_mask 是一个掩码张量，用于指示哪些位置需要被注意，哪些位置是填充的。
        '''
        pooler_out = bert_out['pooler_output']
        #调用这个 BERT 模型的前向传播将返回一个字典，其中包含了多个输出，其中一个是 pooler_output，表示经过池化操作后的输出特征
        return self.trans(pooler_out)   
    '''
        这个池化后的特征被传递给转换层 self.trans，其中会执行丢弃操作、线性转换操作和 ReLU 激活操作，最终得到模型的输出。
        综上所述，这个模型通过 BERT 模型获取句子特征，然后通过线性转换对这些特征进行处理，从而获得最终的输出。
        '''


class ImageModel(nn.Module):
    '''
    该模型基于预训练的 ResNet-50 模型进行特征提取，并对提取的特征进行线性转换以获得最终输出。
    '''
    def __init__(self, config):
        super(ImageModel, self).__init__()
        self.full_resnet = resnet50(pretrained=True)    #在模型初始化时，它创建了一个完整的 ResNet-50 模型，并加载了预训练权重。
        self.resnet = nn.Sequential(
            *(list(self.full_resnet.children())[:-1]),
            nn.Flatten()
        )
        '''
        使用了 nn.Sequential 定义了一个序列模型 self.resnet
        它从完整的 self.full_resnet 中选择了所有子模块（除了最后一个全连接层），然后添加了一个扁平化层 nn.Flatten()
        self.resnet 模型会将输入图像经过 ResNet 的卷积层和池化层处理，并将最终特征扁平化为向量。
        '''

        self.trans = nn.Sequential(
            nn.Dropout(config.resnet_dropout),
            nn.Linear(self.full_resnet.fc.in_features, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )
        '''
        定义了一个线性转换层 self.trans，它由丢弃层、线性层和 ReLU 激活函数组成。
        线性层的输入维度为 ResNet 模型最后一个全连接层的输出维度（即 self.full_resnet.fc.in_features）
        输出维度为 config.middle_hidden_size。
        '''
        
        # 是否进行fine-tune
        for param in self.full_resnet.parameters():
            if config.fixed_image_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, imgs):
        feature = self.resnet(imgs)

        return self.trans(feature)
    '''
        forward 方法接受输入的图像数据 imgs，然后通过 self.resnet 进行特征提取，得到特征向量 feature。
        最后，将这个特征向量输入到 self.trans 进行线性转换，并将转换后的结果作为模型的最终输出。
        '''


class FuseModel(nn.Module):
    '''
    这个融合模型将输入的文本和图像数据分别通过各自的子模型处理，然后在特征层面进行注意力融合，并最终通过全连接分类器进行分类预测。
    '''
    def __init__(self, config):
        super(FuseModel, self).__init__()
        # text
        self.text_model = TextModel(config)
        # image
        self.img_model = ImageModel(config)
        # attention
        self.attention = nn.TransformerEncoderLayer(
            d_model=config.middle_hidden_size * 2,
            nhead=config.attention_nhead, 
            dropout=config.attention_dropout
        )
        '''
        定义了一个注意力模块 self.attention，它是一个基于 Transformer 的编码层。
        输入维度是 config.middle_hidden_size * 2，即文本模型和图像模型的特征维度的两倍。
        nhead 表示注意力头的数量，dropout 是注意力层的丢弃概率。
        '''

        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels)
        )
        '''
        定义了一个全连接分类器 self.classifier，它由一些线性层、激活函数和丢弃层组成
        将输入特征维度 config.middle_hidden_size * 2 转换为 config.num_labels，即最终分类的类别数量。
        同时，定义了交叉熵损失函数 self.loss_func，用于训练时计算损失。
        '''
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, texts, texts_mask, imgs, labels=None):
        text_feature = self.text_model(texts, texts_mask)

        img_feature = self.img_model(imgs)
        '''
        在方法中，首先将文本数据和图像数据分别输入到 self.text_model 和 self.img_model 子模型中，得到相应的特征。
        '''

        attention_out = self.attention(torch.cat(
            [text_feature.unsqueeze(0), img_feature.unsqueeze(0)],
        dim=2)).squeeze()
        '''
        将文本特征和图像特征连接在一起，并通过 self.attention 进行注意力融合，得到 attention_out。
        '''
        prob_vec = self.classifier(attention_out)
        pred_labels = torch.argmax(prob_vec, dim=1)
        '''
        将融合后的特征输入到全连接分类器 self.classifier，得到分类概率向量 prob_vec，然后使用 torch.argmax 得到预测标签。
        '''
        if labels is not None:
            loss = self.loss_func(prob_vec, labels)
            return pred_labels, loss
        else:
            return pred_labels
        '''
        如果 labels 不为 None，则计算损失并返回预测标签和损失值；否则，仅返回预测标签。
        综上所述，这个 Model 模型将文本和图像特征进行融合，并通过注意力机制和全连接分类器来进行多分类预测。
        '''
