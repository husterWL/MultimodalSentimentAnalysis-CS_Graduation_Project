# Muiltimodal_Sentiment_Analysis笔记

## 数据预处理

### Dataloader：

是Pytorch中用来处理模型输入数据的一个工具类；其参数主要包括epoch、integration、batch_size、dataset、shuffle

在定义test_loader时，设置了batch_size=4，表示一次性从数据集中取出4个数据 



#### 数据预处理：

对于MVSA-single数据集：

​	删除数据集中情感正负极性相反的图像文本对，剩下的图像文本对中若图像或者文本有一方的情感极性为中性，则取另一方的情感极性作为该图像文本对的情感标签。若都为中性，则标签为中性。

对于MVSA-multiple数据集：

​	采用投票机制，即有2个或者2个以上的情感标注一致，则保留该图像文本对，否则删除。