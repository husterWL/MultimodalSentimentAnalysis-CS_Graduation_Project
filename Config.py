import os

class config:         #用来配置各个模块的参数
    # 根目录
    root_path = os.getcwd()     #返回当前进程的工作目录
    data_dir = "D:/BaiduNetdiskDownload/MVSA/data"
    train_data_path = os.path.join(root_path, 'Data/train.json')
    test_data_path = os.path.join(root_path, 'Data/test.json')
    # output_path = os.path.join(root_path, 'output')
    output_path = "D:/BaiduNetdiskDownload/output"
    output_test_path = os.path.join(output_path, 'test_out.txt')
    load_model_path = None

    # 一般超参
    epoch = 20
    learning_rate = 3e-5
    weight_decay = 0
    num_labels = 3
    loss_weight = [1.68, 9.3, 3.36]

    # Fuse相关
    fuse_model_type = 'MultiAttention'
    only = None
    middle_hidden_size = 1024
    attention_nhead = 8
    attention_dropout = 0.4
    fuse_dropout = 0.3
    out_hidden_size = 128
    # out_hidden_size = 1024

    # BERT相关
    # fixed_text_model_params = False
    fixed_text_model_params = True
    # bert_name = 'bert-base-uncased'
    bert_name = 'roberta-base'
    # bert_name = 'bert-base-uncased'
    bert_learning_rate = 5e-6
    bert_dropout = 0.2
    shared_size = 1024

    # ResNet相关
    fixed_img_model_params = False
    image_size = 224
    fixed_image_model_params = True
    resnet_learning_rate = 5e-6
    resnet_dropout = 0.2
    # img_hidden_seq = 64
    img_hidden_seq = 32


    # Dataloader params
    checkout_params = {'batch_size': 4, 'shuffle': False}
    train_params = {'batch_size': 8, 'shuffle': True, 'num_workers': 2}    #原batch_size=16
    val_params = {'batch_size': 8, 'shuffle': False, 'num_workers': 2}     #原batch_size=16
    test_params =  {'batch_size': 8, 'shuffle': False, 'num_workers': 2}

    
    