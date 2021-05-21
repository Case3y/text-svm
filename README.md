# text-svm：新闻文本标题分类

## 运行环境
- Python3
- pandas
- jieba
- sklearn

## 数据集下载

本项目使用搜狗实验室开放的搜狐新闻数据(SogouCS)的完整包，下载地址：[搜狐新闻数据](https://www.runoob.com)，下载后将文件重命名为：```news_sohusite_xml.txt``` 放入项目文件夹中

## 项目运行

- 运行 ```word_split.py``` 文件对数据进行筛选预处理，并且输出训练集和测试集的文件

- 运行 ```svm.py``` 文件对训练集和测试机利用 TF-IDF 进行文本特征提取，并且加载 SVM 模型进行学习
