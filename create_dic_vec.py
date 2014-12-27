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
from sklearn.cross_validation import train_test_split 

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


def filter_dictionary(dictionary):
    '''
    低頻度と高頻度のワードを除く感じで
    '''
    dictionary.filter_extremes(no_below=1, no_above=0.2)  # この数字はあとで変えるかも
    return dictionary

def get_vector(dictionary, content):
    '''
    学習用のデータをベクトルに変換する
    '''
    dense_list=[]
    for item in content:
        tmp = dictionary.doc2bow(item)
        dense = list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])
        dense_list.append(dense)
    #次元削除        
    return dense_list

def get_vector2(dictionary, content):
    '''
    テスト用のデータをベクトルに変換す
    '''
    test_dense_list=[]
    for item in content:
        tmp = dictionary.doc2bow(item)
        dense = list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])
        test_dense_list.append(dense)
    
    return test_dense_list

                
def train(dense_list,test_dense):
    '''
    データの学習と検証
    '''
    #学習用のデータ
    data_train=[]
    data_train=dense_list
    #ラベル
    label_train = ["bon","sadao"]
    estimator = RandomForestClassifier()
    # 学習させる
    estimator.fit(data_train, label_train)
    #順番に判定していく。test_dense＝テスト用のデータのベクトルリスト
    for item in test_dense:

        label_predict = estimator.predict_proba(item)
        print (label_predict)


def test_data(contents,words):

    test_words=get_words(contents)
    
    words.append(test_words[0])

    return words


#辞書の作成とテスト用データを名詞分解する
def create_test_dic(test_file_name,noun_words):
    ret={}
    tweet_list=[]

    #テスト用データを一行ずつ読む
    f = open(test_file_name)
    sadao2 = f.readlines()
    f.close()

    i = 0
    #テスト用データを一行ずつ処理していく
    for item in sadao2:
        ret={}
        #ツイートをdicにいれる
        ret[i]=item
        
        #名詞に分解する
        
        tmp = get_words(ret)
        #名詞に分解したものをまとめる
        
        tweet_list.append(get_words(ret))
        #学習用データの名詞リストにテストデータの名詞リストも追加する
        noun_words.append(tmp[0])
        
        ret=""
        i=i+1
    
    #辞書作成
    dictionary = filter_dictionary(corpora.Dictionary(noun_words))
    dictionary.save_as_text("test2.txt")
    
    return [dictionary,tweet_list]


def create_test_data(noun_words):
    
    #辞書の作成とテスト用の名詞リスト作成
    dictionary,test_words = create_test_dic("sadao_test.txt",noun_words);

    # 保存しておく
    test_dense=[]
    
    #テスト用データを名詞に変換していく
    for item in test_words:
        if len(item)>0:
            test_dense.append(get_vector2(dictionary, item))
    return [dictionary,test_dense]
    
#学習用テキストを名詞に分解
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

        train_words = list(words)

        #テスト用の特徴語リストを作成
        dictionary,test_dense = create_test_data(train_words)

        #学習用の特徴語のリストを作成
        dense_list = get_vector(dictionary,words)

        #学習＆検証
        train(dense_list,test_dense)
        
        
    else:
        # 通常はファイルから読み込むだけにする
        dictionary = corpora.Dictionary.load_from_text(file_name)

    #return dictionary


if __name__ == '__main__':
    get_dictionary(create_flg=True)
