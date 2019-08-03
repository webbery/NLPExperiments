import os
import math
from collections import defaultdict
import sys
sys.path.append("..")
import cut_sentence
import json
import time

max_sample=float('inf')
def load_dataset(rootdir):
    # {class: list of documents}
    documents={ }
    rootdir = os.path.abspath(rootdir)+'/'
    directories = os.listdir(rootdir)
    for clazz in directories:
        train_set = open(rootdir+clazz+'/train_set','r',encoding='utf-8')
        print("Load ",clazz)
        document = []
        sample_num = 0
        lines = train_set.readlines()
        for line in lines:
            data = json.loads(line)
            text = cut_sentence.segment(data['text'],type='arr')
            document.append(text)
            sample_num+=1
            if sample_num>max_sample: break
        documents[clazz]=document
        train_set.close()
    return documents

def train_naive_bayes(documents):
    vocabularies=defaultdict(int)
    total_doc_count = 0
    for c,value in documents.items():
        for document in value:
            total_doc_count +=1
            for word in document:
                vocabularies[word]+=1

    print('V: ',len(vocabularies))
    vocabularies = sorted(vocabularies.items(),key=lambda item: item[1],reverse=True)[10000:]
    total_class_count = len(documents)

    prob_class={}
    for key,value in documents.items():
        ''' $log(N_c/N_doc)$ '''
        prob_class[key] = math.log(len(value)/total_doc_count)
        print('log(N_c/N_doc) ',key ,prob_class[key])
    
    words={}
    for word,cnt in vocabularies:
        for c,value in documents.items():
            if not words.__contains__(c):
                words[c] = defaultdict(int)
            count_w_c = 0
            for document in value:
                count_w_c += document.count(word)
            words[c][word]=count_w_c            ## count(w,c)
            # print(word, c, count_w_c)

    likelihood = {}
    for c,value in documents.items():
        likelihood[c]=defaultdict(float)
        count_of_word=0
        for word,cnt in vocabularies:               ## sum of count(w,c)
            count_of_word+=(words[c][word]+1)
        for word,cnt in vocabularies:
            ''' log((count(w,c)+1)/\sum{(count(w',c)+1)}) '''
            likelihood[c][word] = math.log((words[c][word]+1)/count_of_word)
        print(c,' likelihood finish')
    return prob_class,likelihood

def save_model(file_name,logprior,likelihood):
    f = open(file_name,'w',encoding='utf-8')
    json.dump([logprior,likelihood],f)
    f.close()

def load_model(file_name):
    f=open(file_name,'r',encoding='utf-8')
    model = json.load(f)
    f.close()
    return model

def predict_naive_bayes(doc,model):
    logprior = model[0]
    likelihood = model[1]
    # print(doc)

    last_sum = -float('inf')
    doc_class = ''
    for clszz,value in logprior.items():
        sum_c = value
        likelihood_c = likelihood[clszz]
        # start = time.clock()
        for word in doc:
            # print(word)
            if likelihood_c.__contains__(word):
                sum_c += likelihood_c[word]
        if sum_c > last_sum:
            doc_class = clszz
            last_sum = sum_c
            # print(doc_class,last_sum)
        # end = time.clock()
        # print (str(end-start))
    return doc_class

''' 测试数据 '''
def test_naive_bayes(test_file):
    test_set = open(test_file,'r',encoding='utf-8')
    print("Load ",test_file)
    document = []
    for line in test_set.readlines():
        data = json.loads(line)
        text = cut_sentence.segment(data['text'],type='arr')
        document.append(text)
    test_set.close()
    # for doc in document:
    #     yield doc
    return document[3]

def predict_document(file_name,model):
    f=open(file_name,'r',encoding='utf-8')
    document = []
    for line in f.readlines():
        text = cut_sentence.segment(line,type='arr')
        document.append(text)
    f.close()
    return predict_naive_bayes(document,model)

