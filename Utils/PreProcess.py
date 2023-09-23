'''
对于MVSA-single数据集：

	删除数据集中情感正负极性相反的图像文本对，剩下的图像文本对中若图像或者文本有一方的情感极性为中性，则取另一方的情感极性作为该图像文本对的情感标签。若都为中性，则标签为中性。

对于MVSA-multiple数据集：

	采用投票机制，即有2个或者2个以上的情感标注一致，则保留该图像文本对，否则删除。

'''
'''
思路：
    读取文件；循环判断；把通过的图文对写入新文件，新文件格式按照guid，label来
'''
#函数
def majority_vote(sentiments):
    if sentiments[0] == sentiments[1]:
        if sentiments[0][0] == sentiments[0][1]:
            return sentiments[0][0]
        elif sentiments[0][0] + sentiments[0][1] == 0:
            return None
        else:
            return sentiments[0][0] + sentiments[0][1]
    elif sentiments[0] == sentiments[2]:
        if sentiments[0][0] == sentiments[0][1]:
            return sentiments[0][0]
        elif sentiments[0][0] + sentiments[0][1] == 0:
            return None
        else:
            return sentiments[0][0] + sentiments[0][1]
    elif sentiments[1] == sentiments[2]:
        if sentiments[1][0] == sentiments[1][1]:
            return sentiments[1][0]
        elif sentiments[1][0] + sentiments[1][1] == 0:
            return None
        else:
            return sentiments[1][0] + sentiments[1][1]
    else:
        return None 

#读取文件
file = open("D:/BaiduNetdiskDownload/MVSA/labelResultAll.txt", mode='r', encoding="UTF-8")
data = file.readlines()

#判断
with open("D:/BaiduNetdiskDownload/MVSA/train_multi.txt", mode = 'a', encoding = 'UTF-8') as t:
    t.write('guid' + ' ' + 'label\n')
for line in data[1:]:
    parts = line.split()
    ID = parts[0]
    # print('ID:', ID)
    words = parts[1:]
    sentiments = []
    for word in words:
        word_parts = word.split(',')
        #构建一个列表
        '''
        for word_part in word_parts:
            if word_part == 'negative':
                word_part = -1
            elif word_part == 'neutral':
                word_part = 0
            else: 
                word_part = 1
        '''#在这里只是用word_part替换原来的值，但word_parts并没有改变
        #需要使用索引进行修改
        for i in range(len(word_parts)):  
            if word_parts[i] == 'negative':  
                word_parts[i] = -1  
            elif word_parts[i] == 'neutral':  
                word_parts[i] = 0  
            else:   
                word_parts[i] = 1
        sentiments.append(word_parts)
    label = majority_vote(sentiments)
    if label == None:
        continue
    if label == -1:
        label = 'negative'
    elif label == 0:
        label = 'neutral'
    else:
        label = 'positive'
    with open("D:/BaiduNetdiskDownload/MVSA/train_multi.txt", mode = 'a', encoding = 'UTF-8') as t:
        t.write(ID + ' ' + label + '\n')
file.close()