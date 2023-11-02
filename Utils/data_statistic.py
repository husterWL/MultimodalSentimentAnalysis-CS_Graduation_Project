with open("Data/MVSA_pre.txt", mode = 'r', encoding = 'UTF-8') as file:
    lines = file.readlines()
    positive = []
    neutral = []
    negative = []
    for line in lines[1:]:
        # print(line)
        parts = line.split(',')
        ID = parts[0]
        sentiment = parts[1]
        if sentiment == 'positive\n' :
            positive.append(ID)
        elif sentiment == 'neutral\n' :
            neutral.append(ID)
        else:
            negative.append(ID)
    print('positive:' + str(len(positive)) + '\n')
    print('neutral:' + str(len(neutral)) + '\n')
    print('negative:' + str(len(negative)) + '\n')



'''
下面注释代码是因为在运行上述代码时发现有换行符的影响，浴室去运行之前预处理数据的代码，发现没什么问题，可能是因为预处理代码中写文件时每一行多写了一个换行符的原因
'''
# file = open("D:/BaiduNetdiskDownload/MVSA/labelResultAll.txt", mode='r', encoding="UTF-8")
# data = file.readlines()
# for line in data[1:]:
#     parts = line.split()
#     ID = parts[0]
#     print('ID:', ID)
#     words = parts[1:]
#     sentiments = []
#     for word in words:
#         word_parts = word.split(',')
#         #构建一个列表
#         '''
#         for word_part in word_parts:
#             if word_part == 'negative':
#                 word_part = -1
#             elif word_part == 'neutral':
#                 word_part = 0
#             else: 
#                 word_part = 1
#         '''#在这里只是用word_part替换原来的值，但word_parts并没有改变
#         #需要使用索引进行修改
#         for i in range(len(word_parts)):  
#             if word_parts[i] == 'negative':  
#                 word_parts[i] = -1  
#             elif word_parts[i] == 'neutral':  
#                 word_parts[i] = 0  
#             else:   
#                 word_parts[i] = 1
#         print(word_parts)
#     print('\n')