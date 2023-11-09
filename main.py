#训练与测试_主体

import os
# os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

'''
在导入模块时，Python会查找sys.path中列出的目录。默认情况下，Python会将当前目录添加到sys.path中。
如果你在不同的目录中运行你的代码，你可能需要手动添加包含你的模块的目录到sys.path中。
'''
import sys
sys.path.append('./Utils')
# sys.path.append('./Utils/APIs')
import torch
import argparse
from Config import config
from Utils.common import data_format, read_from_file, train_val_split, save_model, write_to_file, loss_draw, acc_draw, macro_draw
from Utils.DataProcess import Processor
from Trainer import Trainer
import matplotlib.pyplot as plt

#消除警告信息。警告信息说明对应的加载的预训练模型与任务类型不完全对应。
from transformers import logging
logging.set_verbosity_error()

# args 参数
'''
argparse模块是Python标准库中的一个模块，它的主要作用是处理命令行参数。
通过使用argparse模块，开发者可以指定脚本需要的参数，包括参数的类型、名字、缩写、数据类型、描述信息等。
这样，用户就可以通过命令行直接为脚本指定参数，而无需在脚本内部修改参数。
'''
parser = argparse.ArgumentParser()
parser.add_argument('--do_train', action='store_true', help='训练模型')
'''
这行代码的作用就是添加一个名为 --do_train 的可选参数，如果用户在命令行中指定了这个参数，
那么 do_train 就会被设为 True，否则就是 False。同时，还提供了帮助信息，方便用户了解这个参数的作用。
'''
'''
在命令行中使用这些参数时，可以使用"--参数名"的形式来指定参数的值。
例如，如果要设置学习率为0.001，可以在命令行中使用"--lr 0.001"来指定。
'''
parser.add_argument('--text_pretrained_model', default='roberta-base', help='文本分析模型', type=str)
parser.add_argument('--fuse_model_type', default='MultiAttention', help='融合模型类别', type=str)
parser.add_argument('--lr', default=5e-5, help='设置学习率', type=float)
parser.add_argument('--weight_decay', default=1e-4, help='设置权重衰减', type=float)
parser.add_argument('--epoch', default=1, help='设置训练轮数', type=int)

parser.add_argument('--do_test', action='store_true', help='预测测试集数据')
# parser.add_argument('--load_model_path', default=None, help='已经训练好的模型路径', type=str)
parser.add_argument('--text_only', action='store_true', help='仅用文本预测')
parser.add_argument('--img_only', action='store_true', help='仅用图像预测')
args = parser.parse_args()
config.learning_rate = args.lr
config.weight_decay = args.weight_decay
config.epoch = args.epoch
config.bert_name = args.text_pretrained_model
config.fuse_model_type = args.fuse_model_type
# config.load_model_path = args.load_model_path
config.load_model_path = os.path.join('d:/BaiduNetdiskDownload/output', args.fuse_model_type, './pytorch_model.bin')
config.only = 'img' if args.img_only else None
config.only = 'text' if args.text_only else None
if args.img_only and args.text_only: config.only = None
print('TextModel: {}, ImageModel: {}, FuseModel: {}'.format(config.bert_name, 'ResNet50', config.fuse_model_type))


# Initilaztion 初始化 在这里选择模型
processor = Processor(config)   #Processor类的实例对象
if config.fuse_model_type == 'CMAC' or config.fuse_model_type == 'CrossModalityAttentionCombine':
    from Model.CMACModel import FuseModel
elif config.fuse_model_type == 'HSTEC' or config.fuse_model_type =='HiddenStateTransformerEncoder':
    from Model.HSTECModel import FuseModel
elif config.fuse_model_type == 'OTE' or config.fuse_model_type == 'OutputTransformerEncoder':
    from Model.OTEModel import FuseModel
elif config.fuse_model_type == 'NaiveCat':
    from Model.NaiveCatModel import FuseModel
elif config.fuse_model_type == 'NaiveCombine':
    from Model.NaiveCombineModel import FuseModel
else:
    from Model.BERT_RESNET_SA import FuseModel
# from Model.CMACModel import FuseModel
# from Model.BERT_RESNET_SA import FuseModel
model = FuseModel(config)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
trainer = Trainer(config, processor, model, device) #实例化了一个训练器，包括配置、预处理、模型和硬件加速


# Train 训练
def train():
    # data_format(os.path.join(config.root_path, './Data/train.txt'),     #join函数是路径拼接作用；数据格式转换成json;./表示的是相对于当前工作目录
    # os.path.join(config.root_path, './Data/data'), os.path.join(config.root_path, './Data/train.json')) #输入地址；数据目录；输出地址
    # data_format(os.path.join(config.root_path, './Data/train.txt'),
    # "D:/BaiduNetdiskDownload/MVSA/data", os.path.join(config.root_path, './Data/train.json'))   #第一次已经格式化为json，后面运行可以跳过
    data = read_from_file(config.train_data_path, config.data_dir, config.only) #训练数据路径。data是json格式的
    train_data, val_data = train_val_split(data)    #将分成训练集和验证集，此时是四元组形式
    train_loader = processor(train_data, config.train_params)   #调用_call_函数，返回to_loader对象；to_loader又使用到了to_dataset。。
    #最终是返回了一个Dataloader对象，注意上面的配置里有train_params
    val_loader = processor(val_data, config.val_params)

    best_acc = 0    #用于记录模型在验证集上的最佳准确率
    #在每个epoch结束后，模型将在验证集上进行评估，并计算其准确率。
    epoch = config.epoch    #20，用于记录当前训练的轮数
    tloss_list, vloss_list = [], []
    acc_list = []
    macro_p = []
    macro_r = []
    macro_f1 = []
    x = range(0, epoch)
    for e in range(epoch):  #左闭右开0-19
        print('-' * 20 + ' ' + 'Epoch ' + str(e+1) + ' ' + '-' * 20)    #打印当前训练的轮数
        tloss, tlosslist = trainer.train(train_loader) #参数是一个Dataloader实例对象，用train函数进行训练，返回训练损失和损失列表
        print('Train Loss: {}'.format(tloss))
        vloss, vacc, report_dict = trainer.valid(val_loader) #valid()函数用于评估模型，并返回验证损失和验证准确率
        print('Valid Loss: {}'.format(vloss))
        print('Valid Acc: {}'.format(vacc))

        tloss_list.append(tloss)
        vloss_list.append(vloss)
        acc_list.append(report_dict['accuracy'])
        macro_p.append(report_dict['macro avg']['precision'])
        macro_r.append(report_dict['macro avg']['recall'])
        macro_f1.append(report_dict['macro avg']['f1-score'])
        # print('accuracy:{}'.format(report_dict['accuracy']))
        '''
        当每次验证准确率高于最佳准确率时都会更新最佳准确率，并且保存模型
        '''
        if vacc > best_acc:
            best_acc = vacc
            save_model(config.output_path, config.fuse_model_type, model)   #保存训练好的模型
            print('Update best model!')
        print()
    #损失曲线
    loss_draw(tloss_list, vloss_list, x, os.path.join(config.output_path, config.fuse_model_type, 'loss_curve.jpg'))

    #准确率曲线
    acc_draw(acc_list, x, os.path.join(config.output_path, config.fuse_model_type, 'accuracy_curve.jpg'))

    #macro曲线
    macro_draw(macro_p, macro_r, macro_f1, x, os.path.join(config.output_path, config.fuse_model_type, 'macro_curve.jpg'))


# Test 测试
def test():
    data_format(os.path.join(config.root_path, './Data/test_without_label.txt'), 
    "D:/BaiduNetdiskDownload/MVSA/data", os.path.join(config.root_path, './Data/test.json'))
    test_data = read_from_file(config.test_data_path, config.data_dir, config.only)
    test_loader = processor(test_data, config.test_params)

    if config.load_model_path is not None:
        model.load_state_dict(torch.load(config.load_model_path))   #只加载参数字典给model，即最前面的实例对象等价于：model.load_state_dict(torch.load(config.load_model_path),model)
        '''
        我觉得应该是这句:
        '''
        #trainer.model.load_state_dict(torch.load(config.load_model_path))
    '''
    load_model_path是要加载的指定预训练模型的存储路径
    若不为空，则使用torch.load()函数加载模型的状态字典，再使用load_state_dict()函数将状态字典加载在模型中
    状态字典包括模型的权重、偏置、学习率等等
    '''

    outputs = trainer.predict(test_loader)
    formated_outputs = processor.decode(outputs)
    write_to_file(config.output_test_path, formated_outputs)


# main
if __name__ == "__main__":
    if args.do_train:
        train()
    
    if args.do_test:
        # if args.load_model_path is None and not args.do_train:
        if config.load_model_path is None and not args.do_train:
            print('请输入已训练好模型的路径load_model_path或者选择添加do_train arg')
        else:
            test()