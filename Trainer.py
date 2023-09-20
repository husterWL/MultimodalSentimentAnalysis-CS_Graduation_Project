import torch
from torch.optim import AdamW
from tqdm import tqdm


class Trainer():            #训练器

    def __init__(self, config, processor, model, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.config = config
        self.processor = processor
        self.model = model.to(device)
        self.device = device
       
        bert_params = set(self.model.text_model.bert.parameters())  
        #set用于建立一个无序的不可重复的集合，存储唯一的元素
        #在这里set将parameters()函数返回的模型参数列表转化为一个集合，以便快速找到其中唯一的参数
        '''self.model.text_model.bert.parameters()返回self.model.text_model.bert模型中的所有参数。
        如果您不使用set将其转换为集合，而是直接使用列表，则可能会得到重复的参数，这将导致错误。
        通过使用set，您可以确保参数列表中没有重复的元素，并可以更轻松地处理和管理参数。'''

        resnet_params = set(self.model.img_model.full_resnet.parameters())
        other_params = list(set(self.model.parameters()) - bert_params - resnet_params)
        '''
        list是一个基本数据类型，用于存储和处理各种类型的数据。即列表，可以将集合转换为列表
        '''
        no_decay = ['bias', 'LayerNorm.weight']
        '''
        是一个列表，用于指定不应该随着训练过程进行正则化的参数。no_decay中的参数不会被正则化，而其他参数将被正则化。
        bias是一个参数，用于调整模型的偏差.
        LayerNorm.weight是一个参数，用于调整模型的层标准化器的权重。这些参数通常不需要正则化，因为它们在训练过程中不会改变。
        '''
        params = [      #named_parameters()函数也是返回参数列表，只不过是按名称顺序返回
            {'params': [p for n, p in self.model.text_model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': self.config.bert_learning_rate, 'weight_decay': self.config.weight_decay},    #正则化
                #p指的是参数的值，n是参数的名称
            {'params': [p for n, p in self.model.text_model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': self.config.bert_learning_rate, 'weight_decay': 0.0},     #不进行正则化
            {'params': [p for n, p in self.model.img_model.full_resnet.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': self.config.resnet_learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.img_model.full_resnet.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': self.config.resnet_learning_rate, 'weight_decay': 0.0},
            {'params': other_params,
                'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay},
        ]
        #最后params是一个列表，包含五个字典：分别是文本模型需要正则化的参数和不正则化的参数、图像模型的需要正则化的参数和不正则化的参数、其他参数
        self.optimizer = AdamW(params, lr=config.learning_rate) #AdamW优化器
        '''
            上面params列表中的字典，用于定义模型中要正则化的参数和其相关的参数设置
            包含两个键：params、lr、weight_decay
            params：一个列表，包含模型中要正则化的参数。该列表是通过使用for循环和if语句从self.model.text_model.bert.named_parameters()中
            提取的。if not any(nd in n for nd in no_decay)用于检查参数的名称是否包含no_decay中的任何参数。
            如果参数的名称不包含no_decay中的任何参数，则该参数将被正则化。
            lr：一个浮点数，指定模型中要正则化的参数的学习率。
            weight_decay：一个浮点数，指定模型中要正则化的参数的权重衰减系数。
        '''

    def train(self, train_loader):
        self.model.train()
        '''
        是一个方法，用于将模型设置为训练模式。train()方法是Model类中的一个方法，用于控制模型的训练模式。
        train()方法的作用是将模型设置为训练模式，即将模型的参数更新为最新的梯度。这样可以在训练过程中优化模型的性能和准确性。
        '''
        loss_list = []
        true_labels, pred_labels = [], []

        #如果collate_fn函数没用，则在这里转化为批次
        for batch in tqdm(train_loader, desc='----- [Training] '):
            guids, texts, texts_mask, imgs, labels = batch
            texts, texts_mask, imgs, labels = texts.to(self.device), texts_mask.to(self.device), \
            imgs.to(self.device), labels.to(self.device)    #to()函数用于将这些列表中的元素转换为PyTorch张量，并将其移动到指定的设备上。
            pred, loss = self.model(texts, texts_mask, imgs, labels=labels) #这一行有问题，没有使用到forward函数，或者Model类中包含call
            '''
            这里调用了模型的forward函数
            调用了forward()函数，是pytorch框架的简写，默认调用forward()函数
            '''
            # metric
            loss_list.append(loss.item())
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss = round(sum(loss_list) / len(loss_list), 5)
        return train_loss, loss_list  

    def valid(self, val_loader):
        self.model.eval()
        '''
        eval()方法是model类中的一个方法，用于控制模型的评估模式。eval()方法的作用是将模型设置为评估模式，
        即将模型的参数设置为最新的梯度。这样可以在评估过程中使用模型的最新参数，从而获得更准确的评估结果。
        '''
        val_loss = 0
        true_labels, pred_labels = [], []

        for batch in tqdm(val_loader, desc='\t ----- [Validing] '):
            guids, texts, texts_mask, imgs, labels = batch
            texts, texts_mask, imgs, labels = texts.to(self.device), texts_mask.to(self.device), imgs.to(self.device), labels.to(self.device)
            pred, loss = self.model(texts, texts_mask, imgs, labels=labels)

            # metric
            val_loss += loss.item()
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())
            
        metrics = self.processor.metric(true_labels, pred_labels)
        return val_loss / len(val_loader), metrics
            
    def predict(self, test_loader):
        self.model.eval()
        pred_guids, pred_labels = [], []

        for batch in tqdm(test_loader, desc='----- [Predicting] '):
            guids, texts, texts_mask, imgs, labels = batch
            texts, texts_mask, imgs = texts.to(self.device), texts_mask.to(self.device), imgs.to(self.device)
            pred = self.model(texts, texts_mask, imgs)

            pred_guids.extend(guids)
            pred_labels.extend(pred.tolist())

        return [(guid, label) for guid, label in zip(pred_guids, pred_labels)]