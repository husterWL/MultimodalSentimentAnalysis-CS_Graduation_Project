'''
普通的常用工具

'''

import os
import json
import chardet
import torch
from tqdm import tqdm   #进度条库
from PIL import Image
from sklearn.model_selection import train_test_split

#乱序并分离数据 2023年9月23日  先用这个把整个数据文件分成训练集、验证集和测试集，后面再在main.py中转化成json文件
import random
def split_dataset(file_path, train_ratio):
    #先生成两个文件
    with open('Data/train.txt', mode = 'a', encoding = 'UTF-8') as file:
        file.write('guid' + ',' + 'label\n')
    with open('Data/test.txt', mode = 'a', encoding = 'UTF-8') as file:
        file.write('guid' + ',' + 'label\n')

    # 读取原始数据集  
    with open(file_path, 'r') as file:  
        lines = file.readlines()  
  
    # 打乱数据顺序  
    random.shuffle(lines)  
  
    # 计算每个集合的大小  
    total_size = len(lines) - 1  
    train_size = int(train_ratio * total_size)  
    # val_size = int(val_ratio * total_size)  
    # test_size = int((1 - train_ratio) * total_size)  
  
    # 划分数据集
    train_set = lines[:train_size]  
    # val_set = lines[train_size:train_size+val_size]  
    test_set = lines[train_size:]  

    # 写入划分后的数据集文件  
    with open('Data/train.txt', mode = 'a', encoding = 'UTF-8') as file:  
        # file.writelines('\n'.join(train_set))   #.join()的作用是在每一个列表元素中插入指定的字符，例如这里插入的是换行符
        file.writelines(train_set)
    # with open('val.txt', 'w') as file:  
    #     file.writelines(val_set)  
    with open('Data/test.txt', mode = 'a', encoding = 'UTF-8') as file:  
        file.writelines(test_set)

# 将文本和标签格式化成一个json
def data_format(input_path, data_dir, output_path):
    data = []
    with open(input_path) as f:
        for line in tqdm(f.readlines(), desc='----- [Formating]'):  #desc为传入进度条的文字描述
            guid, label = line.replace('\n', '').split(',')
            text_path = os.path.join(data_dir, (guid + '.txt')) #原始文本数据路径 例如："。。。\data\data\1.txt"
            if guid == 'guid': continue #去除掉第一行的属性名
            with open(text_path, 'rb') as textf:    #以二进制格式打开文件
                text_byte = textf.read() #text_byte是1.txt的二进制格式数据
                encode = chardet.detect(text_byte)  #返回一个字典，该字典包括判断到的编码格式及判断的置信度。
                try:    #可能错误的语句放在try模块中，except模块用来处理异常
                    text = text_byte.decode(encode['encoding']) #encode['encoding']代表字典中encoding键所对应的值，应该是utf-8
                except:                                         #decode以指定的编码格式解码字符串，默认为字符串编码；encode是编码
                    # print('can\'t decode file', guid)
                    try:
                        text = text_byte.decode('iso-8859-1').encode('iso-8859-1').decode('gbk')
                    except:
                        print('not is0-8859-1', guid)
                        continue
            text = text.strip('\n').strip('\r').strip(' ').strip()
            data.append({
                'guid': guid,
                'label': label,
                'text': text
            })
    with open(output_path, 'w') as wf: #将转换好的数据输出到指定的路径文件中
        json.dump(data, wf, indent=4)   #dump：将python中的对象转化成json储存在文件中；output_path为文件名；indent为缩进级别，打印的文件更规范好看


# 读取数据，返回[(guid, text, img, label)]元组列表
def read_from_file(path, data_dir, only=None): #读取json文件
    data = []
    with open(path) as f:
        json_file = json.load(f)
        for d in tqdm(json_file, desc='----- [Loading]'):
            guid, label, text = d['guid'], d['label'], d['text']    #
            if guid == 'guid': continue

            if only == 'text': img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
            else:
                img_path = os.path.join(data_dir, (guid + '.jpg'))  #找到guid对应的图片路径
                # img = cv2.imread(img_path)
                img = Image.open(img_path)  #存储guid对应的图像数据
                img.load()

            if only == 'img': text = ''

            data.append((guid, text, img, label))   #data列表，包括guid与其相对应的文本、图像以及标签
        f.close()
    return data


# 分离训练集和验证集
def train_val_split(data, val_size=0.2):    #数据分离；训练数据80%；测试数据20%
    return train_test_split(data, train_size=(1-val_size), test_size=val_size)


# 写入数据
def write_to_file(path, outputs):   #可以用来输出测试结果
    with open(path, 'w') as f:
        for line in tqdm(outputs, desc='----- [Writing]'):
            f.write(line)
            f.write('\n')
        f.close()


# 保存模型
def save_model(output_path, model_type, model): #在main中参数为 config.output_path和config.fuse_model_type
    output_model_dir = os.path.join(output_path, model_type)    #输出模型的保存目录
    if not os.path.exists(output_model_dir): os.makedirs(output_model_dir)    # 没有文件夹则创建
    
    '''
    这行代码的作用就是判断模型是否有module属性，如果有，就保存model.module，否则就保存模型本身。
    这样可以确保我们总是能够保存实际模型的状态，无论模型是否被封装在容器中。
    '''
    model_to_save = model.module if hasattr(model, 'module') else model     # Only save the model it-self
    output_model_file = os.path.join(output_model_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)
    '''
    pytorch把所有的模型参数用一个内部定义的dict进行保存，自称为“state_dict”。这个所谓的state_dict就是不带模型结构的模型参数
    '''