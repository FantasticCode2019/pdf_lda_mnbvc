from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm
import argparse
import pyLDAvis.sklearn
import jsonlines
import re
import shutil
import os


# 文本预处理函数
def preprocess_text(txt):
    txt = re.sub(r'[®©™]', '', txt)
    words = word_tokenize(txt.lower())
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w.isalnum() and w not in stop_words]
    return words

# 输出每个主题对应词语
def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
    return tword

def move_file_to_current_directory(source_path, destination_path, operation):
    if operation == "copy":  # 复制文件
        shutil.copyfile(source_path, destination_path)
    else: # 移动文件
        shutil.move(source_path, destination_path)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path",
        type=str,
        default='./data/all_metainfo_230916_1.jsonl',
        help="This is original jsonl file",
    )
    parser.add_argument(
        "--operation",
        type=str,
        default='move',
        help="original file is move or copy",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    lcontent = []
    with jsonlines.open(args.source_path, "r") as reader:
        for pdfInfo in tqdm(reader):
            if '/Producer' in pdfInfo.keys():  # '/Creator'
                lcontent.append(preprocess_text(pdfInfo['/Producer']))
    reader.close()

    corpus = []
    for row in lcontent:
        text = " ".join(row)
        corpus.append(text)

    vectorizer = CountVectorizer(ngram_range=(1,2))
    vector = vectorizer.fit_transform(corpus)  # 转化为词频矩阵
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    tfidf_weight = tfidf.toarray()
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    # word = vectorizer.get_feature_names_out()

    # LDA构建
    n_topics = 4
    lda = LatentDirichletAllocation(n_components=n_topics,  # 主题个数
                                    max_iter=5,  # EM算法最大迭代次数
                                    learning_method='online',  # 只在fit方法中使用，总体说来，当数据尺寸特别大的时候，在线online更新会比批处理batch更新快得多
                                    learning_offset=50.,  # 一个（正）参数，可以减轻online在线学习中的早期迭代的负担。
                                    random_state=0
                                    )
    lda.fit(tfidf_weight)  # 拟合
    topicc = lda.components_  # 主题-词项分布
    topics = lda.transform(tfidf_weight)  # 文档-主题分布

    # 每个单词的主题权重值
    id = 0
    theme = []
    for tt_m in topicc:
        tt_dict = [(name, tt) for name, tt in zip(word, tt_m)]
        tt_dict = sorted(tt_dict, key=lambda x: x[1], reverse=True)
        t = []
        for topic_idx, topic in enumerate(tt_dict[:5]):
            t.append(topic[0])
        theme.append(t)
        id += 1

    n_top_words = 25  # 前几个自己指定
    topic_word = print_top_words(lda, word, n_top_words)
    pic = pyLDAvis.sklearn.prepare(lda, vector, vectorizer, sort_topics=False, mds='tsne')
    pyLDAvis.save_html(pic, 'lda_pass4.html')

    with jsonlines.open(args.source_path, "r") as reader:
        for pdfInfo in tqdm(reader):
            if '/Producer' in pdfInfo.keys():  # '/Creator'
                producer = preprocess_text(pdfInfo['/Producer'])
                file_path = pdfInfo['file_name']
                for i in range(len(theme)):
                    for word in theme[i]:
                        if word in producer and os.path.exists(file_path):
                            # 获取源文件的文件名
                            source_file_name = os.path.basename(file_path)
                            # 构造目标文件的路径
                            destination_file_path = os.path.join(os.getcwd() + '/' + str(i) + '/', source_file_name)
                            # 执行移动操作
                            move_file_to_current_directory(file_path, destination_file_path, operation=args.operation)
    reader.close()