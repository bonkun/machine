#coding:utf-8
import os
import sys
import re
from gensim import corpora, matutils
import MeCab
import codecs
import matplotlib
import matplotlib.pyplot as plt
import numpy
from sklearn import decomposition
mecab = MeCab.Tagger('mecabrc')
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier ,GradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn import datasets
from sklearn.cross_validation import cross_val_score
import numpy
import sklearn.decomposition
 

def get_file_content(file_path):
    '''
    1つの記事を読み込み
    '''
    ret = ''
    f = open(file_path,"r")
    
    for row in f:
        ret = ret + row
    f.close()

    return ret


def tokenize(text):
    '''
    とりあえず形態素解析して名詞だけ取り出す感じにしてる
    '''
    node = mecab.parseToNode(text)
    while node:
        if node.feature.split(',')[0] == '名詞':
            yield node.surface.lower()
        node = node.next

def get_words(contents):
    '''
    記事群のdictについて、形態素解析して返す
    '''
    ret = []
    for k, content in contents.items():
        ret.append(get_words_main(content))
    return ret


def get_words_main(content):
    '''
    一つの記事を形態素解析して返す
    '''
    return [token for token in tokenize(content)]


def get_contents(text_title):
    '''
    dictでまとめる
    '''
    ret = {}

    ret['bon']=(get_file_content(text_title))

    return ret


def get_vector(dictionary, content):
    '''
    ある記事の特徴語カウント
    '''
    dense_list=[]
    for item in content:
        tmp = dictionary.doc2bow(item)
        dense = list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])
        dense_list.append(dense)
    #次元削除        
    lsa = TruncatedSVD(1)
    result = lsa.fit_transform(dense_list)
    return result

def get_vector2(dictionary, content):
    '''
    ある記事の特徴語カウント
    '''
    dense_list=[]
    for item in content:
        tmp = dictionary.doc2bow(item)
        dense = list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])
        dense_list.append(dense)
        
    return dense_list
        
def train(dense_list,test_dense):
    '''
    データの学習と検証
    '''
    data_train=[]

    data_train=dense_list
    
    label_train = [1,2]

    estimator = RandomForestClassifier()

    # 学習させる
    estimator.fit(data_train, label_train)
    test_data = test_dense

    label_predict = estimator.predict_proba(test_data)
    print(label_predict)
    
def test_data(contents,words):

    test_words=get_words(contents)
    
    words.append(test_words[0])

    return words

def create_test_data(test_contents):
    
    ret={}
    ret['test']=test_contents
    words = get_words(ret)
    
    # 辞書作成
    dictionary = corpora.Dictionary(words)
    # 保存しておく
    dictionary.save_as_text("test.txt")
    
    test_dense = get_vector2(dictionary, words)
    #次元削除
    lsa2 = TruncatedSVD(7)
    test_dense = lsa2.fit_transform(test_dense)

    return test_dense
    
    
def create_words(text_list):
    
    ret = {}
    i=0
    for item in text_list:
        ret[item] = get_contents(item)
        tmp_words = get_words(ret[item])
        if i==0:
            words_list = tmp_words
            i = i+1
        else:
            words_list.append(tmp_words[0])
    return words_list
        
def get_dictionary(create_flg=True):
    '''
    辞書を作る,特徴語のリストを作る、検証をする
    '''
    if create_flg or not os.path.exists(file_name):
        #データ名
        text_list=["bon.txt","sadao.txt"]

        #テキストを名詞に分解する
        words=create_words(text_list)
        
        # 辞書作成
        dictionary = corpora.Dictionary(words)

        # 保存しておく
        dictionary.save_as_text("dic.txt")

        #特徴語のリストを作成
        dense_list = get_vector(dictionary,words)

        #今回テストで使いたい文章。これがbonのツイートなのか貞夫のツイートなのか判別したい
        test_contents="だめだーー。全くできん。貞夫のパクツイを検知するプログラムを早く完成させないと貞夫が図に乗りまくる"
        
        #テスト用の特徴語リストを作成
        test_dense = create_test_data(test_contents)
        
        #学習＆検証
        train(dense_list,test_dense)
        
        
    else:
        # 通常はファイルから読み込むだけにする
        dictionary = corpora.Dictionary.load_from_text(file_name)

    #return dictionary


if __name__ == '__main__':
    get_dictionary(create_flg=True)
