# Muiltimodal_Sentiment_Analysis笔记

## 数据预处理

### Dataloader：

是Pytorch中用来处理模型输入数据的一个工具类；其参数主要包括epoch、integration、batch_size、dataset、shuffle

在定义test_loader时，设置了batch_size=4，表示一次性从数据集中取出4个数据 