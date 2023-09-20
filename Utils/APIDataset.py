'''
Dataset api: 与api_encode配合, 将api_encode的返回结果构造成Dataset方便Pytorch调用
    tips:
        注意如果数据长度不一需要编写collate_fn函数, 若无则将collate_fn设为None          ♥♥♥♥♥♥♥♥♥♥
'''

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class APIDataset(Dataset):  #继承于Dataset类

    def __init__(self, guids, texts, imgs, labels) -> None: #->None表示构造函数没有返回值
        self.guids = guids
        self.texts = texts
        self.imgs = imgs
        self.labels = labels    #四个列表

    def __len__(self):
        return len(self.guids)  #返回有多少个样本

    def __getitem__(self, index):
        return self.guids[index], self.texts[index], \
               self.imgs[index], self.labels[index] #上面的\是换行，使用反斜杠转义元组里的换行符。因为这四个元素组成了元组，元组不可变
    
    # collate_fn = None
    def collate_fn(self, batch):    #用于将数据集中的每个样本转换为一个批次，以便在训练和测试过程中使用。
        guids = [b[0] for b in batch]
        texts = [torch.LongTensor(b[1]) for b in batch]
        imgs = torch.FloatTensor([np.array(b[2]).tolist() for b in batch]) 
        labels = torch.LongTensor([b[3] for b in batch])

        ''' 处理文本 统一长度 增加mask tensor '''
        texts_mask = [torch.ones_like(text) for text in texts]

        paded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        paded_texts_mask = pad_sequence(texts_mask, batch_first=True, padding_value=0).gt(0)
        
        
        ''' 处理图像 '''

        return guids, paded_texts, paded_texts_mask, imgs, labels   #collate_fn函数返回经过批次处理的数据