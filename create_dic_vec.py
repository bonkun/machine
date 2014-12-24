#coding:utf-8
import os
import sys
import re
from gensim import corpora, matutils
import MeCab
import codecs

codecs.getwriter(sys.stdout.encoding)(sys.stdout, errors='replace')
DATA_DIR_PATH = './text/'
DICTIONARY_FILE_NAME = 'livedoordic.txt'
mecab = MeCab.Tagger('mecabrc')
from sklearn.ensemble import RandomForestClassifier


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
    return dense_list

        
def train(dense_list,test_dense):

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
    
    test_dense = get_vector(dictionary, words)

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
        
def get_dictionary(create_flg=False, file_name=DICTIONARY_FILE_NAME):
    '''
    辞書を作る
    '''
    if create_flg or not os.path.exists(file_name):
         
        text_list=["bon.txt","sadao.txt"]
        
        words=create_words(text_list)
        
        # 辞書作成
        dictionary = corpora.Dictionary(words)

        # 保存しておく
        dictionary.save_as_text("dic.txt")

        #特徴語のリストを作成
        dense_list = get_vector(dictionary,words)

        #今回テストで使いたいデータ
        test_contents="電車でアホそうなやつがONE PIECEの58巻読んでたから「バカな息子をそれでも愛そう…」のシーンでホロリと泣き出すかなと思ってチラチラ見てたら途中で降りてった"
        
        #テスト用の特徴語リスト
        test_dense = create_test_data(test_contents)
        
        #学習＆検証
        train(dense_list,test_dense)
        
        
    else:
        # 通常はファイルから読み込むだけにする
        dictionary = corpora.Dictionary.load_from_text(file_name)

    #return dictionary


if __name__ == '__main__':
    get_dictionary(create_flg=True)
