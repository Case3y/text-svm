import re
import pandas as pd
import jieba

text = open('news_sohusite_xml.txt',encoding='utf-8').read()
#获取类别
pattern = 'http://(.*?).sohu.com'
category = re.findall(pattern,text,re.S)

#获取标题
p_title = r'<contenttitle>(.*?)</contenttitle>'

title = re.findall(p_title,text)




data = {}
data['category'] = category
data['title'] = title


df = pd.DataFrame(data)
# 更换标题
df.replace(['auto','business','it','sports','learning','club.mil.news','yule'],['汽车','财经','IT','体育','教育','军事','娱乐'],inplace = True)


cate = ['汽车','财经','IT','体育','教育','军事','娱乐']

df_labeled = {}
df_labeled['category'] = []

df_labeled = pd.DataFrame(df_labeled)
# 显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
# pd.set_option('mode.chained_assignment', None)
# df_labeled = df[df['category'].str.contains('公益')]



#筛选分类
for t in range(len(cate)):
    #筛选分类
    bool = df['category'].str.contains(cate[t])
    df_bool = df[bool]
    
    if not df_bool['category'].isin(df_labeled['category']).empty:
        df_labeled = pd.concat([df_bool,df_labeled],ignore_index=True)
# print(df_labeled)

# 输出总分类个数
print('文章总数 = %s'% df['category'].count())
for t in range(len(cate)):
    bool = df['category'].str.contains(cate[t])
    df_bool = df[bool]
    #输出被分类的标题
    print("%s = %s"%(cate[t],df_bool['category'].count()))

# 抽取训练样本
train_num = 20000
df_train = df_labeled.sample(n = train_num,axis=0)
# print(df_train)
#抽取测试样本
test_num = 5000
df_test = df_labeled.sample(n = test_num,axis=0)
# print(df_test)



#stop_list:停用词
stop_list=[]

#训练样本文件
train_title = open('train_title.txt',"w",encoding='utf-8')
train_category = open('train_category.txt',"w",encoding='utf-8')
#测试样本文件
test_title = open('test_title.txt',"w",encoding='utf-8')
test_category = open('test_category.txt',"w",encoding='utf-8')

#停用词文件
stopwords = open('hit_stopwords.txt',"r",encoding='utf-8')
for line in stopwords.readlines():
    stop_list.append(line.replace('\n', ''))


#分割后的训练集标题 train_title
tr_t_num = 0
for index,row in df_train.iterrows():
    words = jieba.cut(row['title'])
    for word in words:
        if word not in stop_list:
            train_title.write(word + " ")
    tr_t_num += 1
    if tr_t_num < train_num:
        train_title.write('\n')
train_title.close()

tr_c_num = 0
#训练集类别 train_category
for index,row in df_train.iterrows():
    words = jieba.cut(row['category'])
    for word in words:
        if word not in stop_list:
            train_category.write(word + " ")
    tr_c_num += 1
    if tr_c_num < train_num:
        train_category.write('\n')
train_category.close()

#分割后的测试集标题 test_title
t_t_num = 0
for index,row in df_test.iterrows():
    words = jieba.cut(row['title'])
    for word in words:
        if word not in stop_list:
            test_title.write(word + " ")
    t_t_num += 1
    if t_t_num < test_num:
        test_title.write('\n')        
test_title.close()

#训练集类别 train_category
t_c_num = 0
for index,row in df_test.iterrows():
    words = jieba.cut(row['category'])
    for word in words:
        if word not in stop_list:
            test_category.write(word + " ")
    t_c_num += 1
    if t_c_num < test_num:
        test_category.write('\n')
test_category.close()
