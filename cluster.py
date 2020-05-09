import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import Birch

data = pd.read_excel('附件3.xlsx')
x = data.留言主题

# 停用词
stopwords = []
with open('stopwords.txt', errors='ignore') as sf:
    for line in sf.readlines():
        stopwords.append(line.strip())

# 分词处理
def text_cut(in_text):
    words = jieba.lcut(in_text)
    cut_text = ' '.join([w for w in words if w not in stopwords and len(w) > 1])
    return cut_text

x_change = []
for i in x:
    x_change.append(text_cut(i))

# 特征提取
vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,2), strip_accents='unicode', norm='l2', token_pattern=r"(?u)\b\w+\b")
X = vectorizer.fit_transform(x_change)
num_clusters = 390
birch_cluster = Birch(n_clusters=num_clusters)
birch_result = birch_clusterer.fit_predict(X)
print("Predicting result: ", birch_result)

# 保存
data['类别编号'] = birch_result
pd.DataFrame(data).to_excel('文本数据集.xlsx', sheet_name='Sheet1', index=False, header=True)
