import torch
import torch.nn as nn
from Config import config
from transformers import AutoModel  #可以动态加载预训练模型
from transformers import AutoTokenizer
import torchvision.models as models
#from transformers import BertModel, BertConfig

'''
根据项目的模型图，源码是在各个子模型中得到文本特征、图像特征的
在融合模型中建立注意力机制，得到被关注的特征
然后使用cat连接张量（文本特征与注意力结构输出的特征；图像特征与注意力结构输出的特征），再使用全连接分类器获得概率向量
然后融合两个概率向量并使用softmax进行分类
'''
'''
以我的模型结构进行更改的话
仔细看了模型图，不需要多写一个前向传播
只需要在forward()函数中设置多个输出即可
也就是说在
TextModel中的forward()函数输出语义特征、多级语义特征的联合特征
ImageModel中的forward()函数输出区域特征、多级视觉的联合特征
'''

class TextModel(nn.Module):
    def __init__(self, config):
        super(TextModel,self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.trans = nn.Sequential( 
            nn.Dropout(config.bert_dropout),    #nn.Dropout()是一个nn.Module对象，用于在神经网络模型的训练过程中随机删除一部分神经元，以防止过拟合。
            nn.Linear(self.bert.config.hidden_size, config.middle_hidden_size), #nn.Linear()是一个nn.Module对象，用于将神经网络模型中的输入数据映射为输出数据。
            nn.ReLU(inplace=True)   #nn.ReLU(inplace=True)是一个nn.Module对象，用于将神经网络模型中的输入数据映射为输出数据，并在训练过程中使用 inplace 算法。
        )
        self.word2shared = nn.Linear(self.bert.config.hidden_size, config.shared_size)
        self.phrase2shared = nn.Linear(self.bert.config.hidden_size, config.shared_size)
        self.doc2shared = nn.Linear(self.bert.config.hidden_size, config.shared_size)
        '''
            self.trans 是一个由几个层组成的序列，其中包括一个丢弃层（nn.Dropout）、一个线性层（nn.Linear）和一个 ReLU 激活函数。
            这段代码构建了一个神经网络模块，该模块首先应用丢弃层 (Dropout)，然后通过线性层 (Linear) 执行线性变换
            并最后通过 ReLU 激活函数来处理输出。这种模块通常用于深度学习任务中的特征变换和非线性激活操作。
            '''
        '''
        在上述代码中，self.trans 是一个包含了一系列神经网络层的模块。当你将 local_features 传递给 self.trans(local_features) 时，它的作用是对 local_features 进行一系列操作，包括：

        Dropout 操作：nn.Dropout(config.resnet_dropout) 可以在训练过程中随机丢弃一部分节点的输出，以防止过拟合。

        全连接层操作：nn.Linear(self.full_resnet.fc.in_features, config.middle_hidden_size) 将局部特征映射到一个新的特征空间，其中 config.middle_hidden_size 是新特征的维度。

        激活函数操作：nn.ReLU(inplace=True) 使用 ReLU 激活函数对新特征进行非线性变换。

        这些操作的目的是将 local_features 转换成一个更高级别的表示，以便后续任务（如情感分类）可以更好地利用。这种转换通常有助于学习到更抽象、更有表达力的特征。

        区别在于，经过 self.trans(local_features) 处理后，获得的特征不再具有与原始局部特征相同的语义，而是经过了一系列变换和抽象。这些特征可能更适合你任务的需求，因此通常会在这些高级表示上进行后续的任务（如情感分类）。

        要弄清楚哪种表示更适合你的任务，通常需要通过实验来验证，以确定哪种特征更能提高模型的性能。
        '''
        # 是否进行fine-tune
        for param in self.bert.parameters():
            if config.fixed_text_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def forward(self, bert_inputs, masks, token_type_ids=None):
        # tokenizer = AutoTokenizer.from_pretrained(config.bert_name)
        #修改重点
        bert_out = self.bert(input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)
        
        #获取词向量表示
        #word_vectors = bert_out[0][:,0,:].squeeze()
        word_vectors = bert_out.last_hidden_state
        #获取短语向量表示
        '''
        这里需要做修改
        '''
        #phrase_vectors = torch.mean(word_vectors[tokenizer.wordpiece_tokenizer.tokenize()], dim = 0) 
        phrase_vectors = bert_out.pooler_output
        #获取文档向量表示
        doc_vectors = torch.mean(word_vectors, dim = 1)
        # print(word_vectors.shape)
        # print(phrase_vectors.shape)
        # print(doc_vectors.shape)
        '''
        我觉得还是通过注意力融合比较好，可以经过实验来验证
        '''
        #嵌入公共空间，使用tanh作为映射函数
        # embedding1 = torch.tanh(word_vectors)
        # embedding2 = torch.tanh(phrase_vectors)
        # embedding3 = torch.tanh(doc_vectors)
        phrase_vectors = phrase_vectors.unsqueeze(1).expand(-1, word_vectors.size(1), -1)
        doc_vectors = doc_vectors.unsqueeze(1).expand(-1, word_vectors.size(1), -1)
        embedding1 = self.word2shared(word_vectors)
        embedding2 = self.phrase2shared(phrase_vectors)
        embedding3 = self.doc2shared(doc_vectors)

        '''
        由于句子长度不一，特征维度是不断变化的 54
        torch.Size([16, 54, 768])
        torch.Size([16, 768])
        torch.Size([54, 768])
        '''
        # print(embedding1.shape)
        # print(embedding2.shape)
        # print(embedding3.shape)
        #使用元素相乘获得联合特征
        #需要修改，不能直接进行相乘，可以使用nn.linear()先将三个层次的特征映射到同一个向量空间中
        #RuntimeError: mat1 and mat2 shapes cannot be multiplied (54x768 and 16x768)
        # joint_feature = torch.mm(embedding1,embedding2)*embedding3  #torch.mm()处理的是矩阵乘法，不是哈达玛积
        #RuntimeError: The size of tensor a (52) must match the size of tensor b (16) at non-singleton dimension 0
        # embedding2 = nn.functional.pad(embedding2, (36, 0))
        '''
        填充没有意义，还是在映射的时候没处理好
        '''
        joint_feature = embedding1*embedding2*embedding3
        # print(joint_feature.shape)
        '''
        还得搞清楚这个隐藏层和池化层的作用，哪个是高级语义特征
        前者是词向量特征，后者是短语、句子特征
        '''
        # hidden_state = bert_out['last_hidden_state']
        # print(hidden_state.shape)
        # pooler_out = bert_out['pooler_output']
        
        # return self.trans(joint_feature), self.trans(hidden_state)
        return joint_feature, embedding1


class ImageModel(nn.Module):    #这个地方的改动似乎不多
    
    #获得图像的局部特征与全局特征，再嵌入公共空间兵相乘获得视觉联合特征

    def __init__(self, config):
        super(ImageModel, self).__init__()
        self.full_resnet = models.resnet50(pretrained=True)    #在模型初始化时，它创建了一个完整的 ResNet-50 模型，并加载了预训练权重。
        self.resnet_l = nn.Sequential(
            *(list(self.full_resnet.children())[:-2]),
        )
        
        # self.resnet_p = nn.Sequential(
        #     list(self.full_resnet.children())[-2],
        #     nn.Flatten()
        # )

        #选取的是res3卷积块中的第三个卷积层
        self.resnet_g = nn.Sequential(
            *(list(self.full_resnet.children())[:-1]),
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
        
        #这个地方的config参数需要修改？
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
    
    def forward(self, imgs):
        global_feature = self.resnet_g(imgs)  #全局
        #feature = self.resnet_p(hidden_state)   #局部
        #下面一句：RuntimeError: Given groups=1, weight of size [256, 1024, 1, 1], expected input[16, 3, 224, 224] to have 1024 channels, but got 3 channels instead
        local_feature = self.resnet_l(imgs) #局部
        joint_feature = global_feature * local_feature #联合    torch.Size([16, 64, 1024])
        local_feature = self.hidden_trans(local_feature)    #torch.Size([16, 64, 1024])
        #shape()返回的是一个对象
        # print(global_feature.shape)
        # print(local_feature.shape)
        joint_feature = self.hidden_trans(joint_feature)
        # print(joint_feature.shape)
        '''
        torch.Size([16, 2048, 1, 1])
        torch.Size([16, 2048, 7, 7])
        torch.Size([16, 2048, 7, 7])
        '''
        #RuntimeError: mat1 and mat2 shapes cannot be multiplied (229376x7 and 2048x1024)   
        # return self.trans(joint_feature), self.trans(local_feature)   #trans用于变换，获得不同维度的特征
        return joint_feature, local_feature

#定义注意力机制
class Attention_Visual(nn.Module):                                                
    def __init__(self, config):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(config.shared_size, config.img_hidden_seq, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(config.img_hidden_seq, config.shared_size)
        ) #按维度设定线性层

    def forward(self, text_features, local_features):
        # 计算相关性分数
        '''
        计算文本特征 (text_features) 和局部特征 (local_features) 之间的相关性分数。
        1、 过线性映射操作 self.W，将文本特征 text_features 投影到与局部特征相同的维度空间，以确保它们可以进行点积操作。
            这个映射的目的是使两个特征具有相同的特征维度，以便进行相似度计算。
        2、 local_features.transpose(1, 2)：这一步对局部特征进行转置操作
            将局部特征的维度从 (batch_size, feature_dim, num_local_features) 转置为 (batch_size, num_local_features, feature_dim)
            其中 num_local_features 是局部特征的数量，feature_dim 是特征的维度。这个转置操作是为了使局部特征的维度与文本特征的维度对齐，以便进行点积操作。
        3、 使用 torch.matmul 函数执行矩阵乘法操作。具体来说，它计算了文本特征矩阵和局部特征矩阵的乘积，以获得每个文本特征与每个局部特征之间的相似性分数。
            结果是一个形状为 (batch_size, num_text_features, num_local_features) 的张量，其中 num_text_features 是文本特征的数量，num_local_features 是局部特征的数量。
            这个张量中的每个元素表示一个文本特征与一个局部特征之间的相似性得分。
        '''
        text_features = text_features.view(text_features.size(0), -1, 1, text_features.size(1))
        '''
        local_features = local_features.permute(0, 2, 3, 1).contiguous().view(local_features.size(0), -1, text_features.size(1))
        RuntimeError: permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = 3 is not equal to len(dims) = 4
        '''
        local_features = local_features.permute(0, 2, 3, 1).contiguous().view(local_features.size(0), -1, text_features.size(1))
        attention_weights = self.attention(text_features).view(text_features.size(0), -1, 1, 1)
        attention_weights = attention_weights.expand(-1, -1, local_features.size(2), -1)
        weighted_local_features = (attention_weights * local_features).sum(dim = 1)
        # scores = torch.matmul(self.W(text_features), local_features.transpose(1, 2))
        # 计算注意力权重
        # attention_weights = torch.softmax(scores, dim=2)
        
        # 使用注意力权重融合局部特征
        # weighted_local_features = torch.matmul(attention_weights, local_features)
        
        return weighted_local_features

class Attention_Text(nn.Module):
    def __init__(self, visual_feature_dim, text_feature_dim):
        super(Attention_Text, self).__init__()
        self.W = nn.Linear(visual_feature_dim, text_feature_dim) #按维度设定线性层

    def forward(self, visual_features, text_features):
        # 计算相关性分数
        scores = torch.matmul(self.W(visual_features), text_features.transpose(1, 2))
        
        # 计算注意力权重
        attention_weights = torch.softmax(scores, dim=2)
        
        # 使用注意力权重融合局部特征
        weighted_text_features = torch.matmul(attention_weights, text_features)
        
        return weighted_text_features

class Attention_Fuse(nn.Module):
    def __init__(self, text_feature_dim, local_feature_dim):
        super(Attention_Fuse, self).__init__()
        self.W = nn.Linear(text_feature_dim, local_feature_dim) #按维度设定线性层

    def forward(self, text_features, local_features):
        # 计算相关性分数
        scores = torch.matmul(self.W(text_features), local_features.transpose(1, 2))
        
        # 计算注意力权重
        attention_weights = torch.softmax(scores, dim=2)
        
        # 使用注意力权重融合局部特征
        weighted_local_features = torch.matmul(attention_weights, local_features)
        
        return weighted_local_features

# 定义多模态情感分析模型
class FuseModel(nn.Module):
    def __init__(self, config):
        super(FuseModel, self).__init__()
        #text
        self.text_model = TextModel(config)
        #image
        self.image_model = ImageModel(config)
        '''
        视觉注意力步骤：
        1、 定义注意力机制，用于计算文本和图像之间的相关性权重。
            这个机制将文本联合特征和图像局部特征作为输入，并输出一个权重向量，表示了文本与图像每个局部特征之间的相关性。
        2、 计算注意力权重，使用定义的注意力机制来计算文本与图像局部特征之间的注意力权重。
            这些权重可以反映文本与每个局部特征的相关性，允许模型选择性地关注不同局部特征。
        3、 加权融合，将图像局部特征与计算得到的注意力权重相乘，以获得加权的图像局部特征。
            这将强调与文本相关的局部特征，并减少与文本不相关的部分。
        '''
        #最重要的是张量的形状要对齐
        #attention
        # self.visual_attention = Attention_Visual(config)
        # self.text_attention = Attention_Text(config)
        self.visual_attention = nn.MultiheadAttention(
            embed_dim=config.middle_hidden_size,
            num_heads=config.attention_nhead, 
            dropout=config.attention_dropout,
        )
        self.text_attention = nn.MultiheadAttention(
            embed_dim=config.middle_hidden_size,
            num_heads=config.attention_nhead, 
            dropout=config.attention_dropout,
        )
        self.fuse_attention = nn.MultiheadAttention(
            embed_dim=config.middle_hidden_size,
            num_heads=config.attention_nhead, 
            dropout=config.attention_dropout
        )

        #全连接分类器
        self.text_classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels),
            nn.Softmax(dim=1)
        )
        self.image_classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels),
            nn.Softmax(dim=1)
        )

        #损失函数
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, texts, texts_mask, images, labels=None):
        
        text_joint_feature, text_feature = self.text_model(texts, texts_mask)
        image_joint_feature, image_feature = self.image_model(images)
        # print(text_joint_feature.shape)
        # print(text_feature.shape)
        # print(image_joint_feature.shape)
        # print(image_feature.shape)
        '''
        torch.Size([16, 53, 1024])
        torch.Size([16, 53, 1024])
        torch.Size([16, 64, 1024])
        torch.Size([16, 64, 1024])
        '''
        #RuntimeError: shape '[58, 512, 128]' is invalid for input of size 950272   =16 58 1024
        text_joint_feature = text_joint_feature.permute(1, 0, 2)
        image_joint_feature = image_joint_feature.permute(1, 0, 2)
        image_feature = image_feature.permute(1, 0, 2)
        text_feature = text_feature.permute(1, 0, 2)

        #注意力分数
        visual_attention_out, _ = self.visual_attention(image_feature, text_joint_feature, text_joint_feature)  #torch.Size([64, 16, 1024])
        text_attention_out, _ = self.text_attention(text_feature, image_joint_feature, image_joint_feature) #torch.Size([len, 16, 1024])
        # print(visual_attention_out.shape)
        # print(text_attention_out.shape)
        # visual_attention_out = visual_attention_net(text_joint_feature, image_feature)
        # text_attention_out = text_attention_net(image_joint_feature, text_feature)

        visual_prob_vec = self.image_classifier(torch.cat([image_feature, visual_attention_out], dim = 2))    #torch.Size([64, 32, 3])
        text_prob_vec = self.text_classifier(torch.cat([text_feature, text_attention_out], dim = 2))  #torch.Size([54, 32, 3])
        # print(visual_prob_vec.shape)
        # print(text_prob_vec.shape)
        visual_prob_vec = torch.mean(visual_prob_vec, dim = 0).squeeze(0)
        text_prob_vec = torch.mean(text_prob_vec, dim = 0).squeeze(0)
        #接下来就是要对两个特征进行模态融合   要修改
        # visual_prob_vec = visual_prob_vec.permute(1, 0, 2)
        # text_prob_vec = text_prob_vec.permute(1, 0, 2)

        #AssertionError: was expecting embedding dimension of 1024, but got 3
        # fused_features, _ = self.fuse_attention(visual_prob_vec, text_prob_vec, text_prob_vec)
        fused_features = visual_prob_vec + text_prob_vec
        prob_vec = torch.softmax(fused_features, dim=1)
        pred_labels = torch.argmax(prob_vec, dim=1)



        if labels is not None:
            #ValueError: Expected input batch_size (32) to match target batch_size (16).
            loss = self.loss_func(prob_vec, labels)
            return pred_labels, loss
        else:
            return pred_labels