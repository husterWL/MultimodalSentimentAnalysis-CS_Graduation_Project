'''
data process: 数据处理, 包括 标签Vocab 和 数据处理类
    tips:
        其中标签Vocab实例化对象必须在api_encode中被调用(add_label)
'''

from torch.utils.data import DataLoader
#用于将数据集划分为批次，并在训练和测试过程中提供这些批次。Dataloader类的主要作用是加速数据的读取和处理，并使数据的读取和处理更加高效和可靠。
from APIDataset import APIDataset
from APIEncode import api_encode
from APIDecode import api_decode
from APIMetric import api_metric


class LabelVocab:
    UNK = 'UNK'

    def __init__(self) -> None:
        self.label2id = {}  #字典
        self.id2label = {}

    def __len__(self):
        return len(self.label2id)

    def add_label(self, label):
        if label not in self.label2id:  #当label2id中没有键label
            self.label2id.update({label: len(self.label2id)})   #update更新字典，更新键值对；相同替换；不相同添加   长度即为id（从0开始）
            self.id2label.update({len(self.id2label): label})

    def label_to_id(self, label):
        return self.label2id.get(label) #返回键值
    
    def id_to_label(self, id):
        return self.id2label.get(id)


class Processor:

    def __init__(self, config) -> None: #初始化实例对象时调用，相当于构造函数
        self.config = config
        self.labelvocab = LabelVocab()
        pass

    def __call__(self, data, params):   #call函数使实例对象能够像函数一样被调用（可调用对象）
        return self.to_loader(data, params)

    def encode(self, data):
        return api_encode(data, self.labelvocab, self.config)
    
    def decode(self, outputs):
        return api_decode(outputs, self.labelvocab)

    def metric(self, inputs, outputs):
        return api_metric(inputs, outputs)
    
    def to_dataset(self, data):
        dataset_inputs = self.encode(data)
        return APIDataset(*dataset_inputs)  #返回一个APIDataset实例对象，并没有对数据做任何操作
    #星号代表可变参数列表，APIDataset(*dataset_inputs)意味着APIDataset类可以
    #接受一个名为dataset_inputs的可变参数列表，并将其展开为单独的参数，然后将这些参数传递给__init__方法中的其他参数。

    def to_loader(self, data, params):
        dataset = self.to_dataset(data) #这里的dataset就是一个APIDataset实例对象
        return DataLoader(dataset=dataset, **params, collate_fn=dataset.collate_fn) #在dataloader中用于将数据集划分为批次；可以加速数据的读取和处理
        #双星号是可变参数字典，一个是可变参数列表
        #返回一个Dataloader对象
        #params包括batch_size、shuffle、num_workers
        '''
        这里的collate_fn应该有参数？如果数据长度一致就可以设置成collate_fn=None
        '''