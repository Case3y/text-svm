from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.svm import SVC

# 加载文件
print('(1) load text...')
# 训练样本文件
train_title = open('train_title.txt',encoding='utf-8').read().split('\n')
train_category = open('train_category.txt',encoding='utf-8').read().split('\n')
# 测试样本文件
test_title = open('test_title.txt',encoding='utf-8').read().split('\n')
test_category = open('test_category.txt',encoding='utf-8').read().split('\n')
all_title = train_title + test_title

# 特征抽取
print ('(2) doc to var...')
count_v0= CountVectorizer()
counts_all = count_v0.fit_transform(all_title)
count_v1= CountVectorizer(vocabulary=count_v0.vocabulary_)
counts_train = count_v1.fit_transform(train_title)
print ("the shape of train is "+repr(counts_train.shape))  
count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_)
counts_test = count_v2.fit_transform(test_title)
print ("the shape of test is "+repr(counts_test.shape))



tfidftransformer = TfidfTransformer();    
train_data = tfidftransformer.fit(counts_train).transform(counts_train);
test_data = tfidftransformer.fit(counts_test).transform(counts_test); 

x_train = train_data
y_train = train_category
x_test = test_data
y_test = test_category


# 加载SVM模型分类
print ('(3) SVM...')
  
svclf = SVC(kernel = 'linear') 
svclf.fit(x_train,y_train) 
preds = svclf.predict(x_test)


for doc, category in zip(y_test, preds):
    print('%r => %s' % (doc, category))

num = 0
for i,pred in enumerate(preds):
    if pred == y_test[i]:
        num += 1
print ('precision_score:' + str(float(num) / len(preds)))





