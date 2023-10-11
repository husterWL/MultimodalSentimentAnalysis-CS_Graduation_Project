'''
encode api: 将原始data数据转化成APIDataset所需要的数据
    tips:
        ! 必须调用labelvocab的add_label接口将标签加入labelvocab字典
'''

from tqdm import tqdm
from transformers import AutoTokenizer
from torchvision import transforms


def api_encode(data, labelvocab, config):

    ''' 这里直接加入三个标签, 后面就不需要添加了 ''' #对应的键值 0；1；2；3
    labelvocab.add_label('positive')
    labelvocab.add_label('neutral')
    labelvocab.add_label('negative')
    labelvocab.add_label('null')    # 空标签

    ''' 文本处理 BERT的tokenizer '''    #构建一个分词器tokenizer("str")会将str分成单个word包括标点，然后再转化成数字；最后返回一个字典
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name)     #使用VPN全局模式，否则连接不到huggingface
    
    ''' 图像处理 torchvision的transforms '''
    def get_resize(image_size):
        for i in range(20):
            if 2**i >= image_size:
                return 2**i
        return image_size
    
    #进行前向传播，获得特征图
    img_transform = transforms.Compose([    #构建一个图像处理器，进行图像预处理操作，对图片进行归一化、大小缩放等等
                transforms.Resize(get_resize(config.image_size)),   #config中的image_size大小为224
                transforms.CenterCrop(config.image_size),   #以输入图的中心点为参考点，按需要的大小进行裁剪
                transforms.RandomHorizontalFlip(0.5),   #用于对载入的数据按随机概率进行水平翻转，垂直反转是Vertical
                transforms.ToTensor(),  #用于对载入的图片数据进行类型转换，转变成张量
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #原始数据的均值与标准差计算出来的，更常用的方法是抽样估计一个标准差与均值
    ])

    ''' 对读入的data进行预处理 '''
    guids, encoded_texts, encoded_imgs, encoded_labels = [], [], [], []
    for line in tqdm(data, desc='----- [Encoding]'):    #data属于四元组形式，即common.py文件中read_from_file函数返回的data
        guid, text, img, label = line
        # id
        guids.append(guid)  #循环拼接，相当于从四元组列表变成四个列表
        
        # 文本
        text.replace('#', '')
        tokens = tokenizer.tokenize('[CLS]' + text + '[SEP]')   #BERT    其实这里可以直接使用tokenizer.encode方法一步到位
        encoded_texts.append(tokenizer.convert_tokens_to_ids(tokens))
        '''
        sentence = "Hello, my son is cuting."
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


        input_ids_method1 = torch.tensor(
            tokenizer.encode(sentence, add_special_tokens=True))  # Batch size 1
            # tensor([ 101, 7592, 1010, 2026, 2365, 2003, 3013, 2075, 1012,  102])

        input_token2 = tokenizer.tokenize(sentence)
            # ['hello', ',', 'my', 'son', 'is', 'cut', '##ing', '.']
        input_ids_method2 = tokenizer.convert_tokens_to_ids(input_token2)
            # tensor([7592, 1010, 2026, 2365, 2003, 3013, 2075, 1012])
            # 并没有开头和结尾的标记：[cls]、[sep]
        '''


        # 图像
        encoded_imgs.append(img_transform(img)) #具体见上方img_transform
        
        # 标签
        encoded_labels.append(labelvocab.label_to_id(label))    #将标签转换成数字，例如positive为0，neutral为1。。。

    return guids, encoded_texts, encoded_imgs, encoded_labels   #返回四个列表

