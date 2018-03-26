#!/usr/bin/env python
# encoding=utf-8

from tqdm import tqdm
import pickle
import glob
import re
import logging
import jieba
import multiprocessing as mp
from multiprocessing import Process, Queue
from collections import Counter
import string
import json
from shuffle_data import shuffle_big_files

en_punctuation = string.punctuation
cn_punctuation = json.load(open("/data/xueyou/textsum/chinese_punctuation.json"))
keeps = '^\u4e00-\u9fa50-9a-zA-Z ' + en_punctuation + cn_punctuation
keeps = u'[{0}]+'.format(keeps)

datetime_pattern = re.compile(r"(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2})|(\d{4}-\d{1,2}-\d{1,2})")
              
question_end = ['?','？']
exclamation_end = ['!','！']
ellipsis_end = ['. . .','…','。 。 。']

black_list = ['关注','公众号','原创','订阅','微信号','来源','蓝字','授权','原创','二维码','下一页']

def predict_style(headline):
    # question
    if headline.endswith(tuple(question_end)):
        return '<style-question>'
    if headline.endswith(tuple(exclamation_end)):
        return '<style-exclamation>'
    if headline.endswith(tuple(ellipsis_end)):
        return '<style-ellipsis>'
    return '<style-declarative>'
    
    
def get_sentences(headline,content, author = None):
    if len(content) >= 100:
        if headline:
            # remove headline
            title_match = content.find(headline.strip())
            if title_match != -1:
                content = content[title_match + len(headline):].strip()
                
            # remove author
            if author:
                author_match = content.rfind(author.strip())
                if author_match != -1:
                    content = content[author_match + len(author):].strip()
            
        dt_match = datetime_pattern.search(content[:100])
        if dt_match:
            content = content[dt_match.end() + 1:]

        # remove black list
        for word in black_list:
            idx = content[:100].rfind(word)
            if idx != -1:
                content = content[idx + len(word):]

        content = re.sub(keeps,'',content)

        if len(content) >= 100:
            sentence = [token.strip().lower() for token in jieba.cut(content) if token and token.strip() and token!='\ufeff'] 
            if sentence:
                yield sentence
                    
def get_headline(t):
    return [token.strip().lower() for token in jieba.cut(t.replace(" ",'')) if token and token.strip()], predict_style(t)
                    

def worker_fashion_web(name,texts):
    print("start to process {0} with len {1}".format(name,len(texts)))
    headlines = []
    articles = []
    cnt = 0
    for item in texts:
        content = []
        headline = item['headline'].strip()
        if headline and len(headline)>=10:
            for s in get_sentences(headline,item['content']):
                content.extend(s)

            if content and len(content)>=50:
                hds, hstyle = get_headline(headline)
                if len(hds) >= 3 and len(hds)<=60:
                    headlines.append(hds)
                    articles.append([hstyle] + content)
            
        cnt += 1
        if cnt % 1000 == 0:
            print("{0} processed {1} articles".format(name,cnt))
            break
    print("finish processing {0} with articles {1}".format(name,len(headlines)))
    return (headlines,articles)

def process_tmall(content):
    refine = []
    content = content.replace("<para>","\n")
    for p in content.split("\n"):
        if p.strip():
            p = p.strip()
            if p.startswith("<ImageItem"):
                tokens = p.split(",")
                if len(tokens) == 2:
                    refine.append(tokens[-1][:-1])
            else:
                refine.append(p)
    return '\n'.join(refine)

def worker_file(fname, n, h, c, a=None, tmall=False):
    print("start to proces file ",fname)
    cnt = 0
    headlines = []
    articles = []
    for line in open(fname):
        tokens = line.strip().split("\x01")
        content = []
        if len(tokens) == n:
            headline = tokens[h].strip()
            # 天猫文章需要特殊处理
            if tmall:
                tokens[c] = process_tmall(tokens[c])
            if headline:
                for x in get_sentences(headline,tokens[c].replace("<para>","\n"), tokens[a] if a else None):
                    content.extend(x)
                # remove content with too few words: 50
                if content and len(content)>=50 :
                    hds, hstyle = get_headline(headline)
                    # remove headline with too few words: 3
                    if len(hds) >= 3 and len(hds)<=60:
                        headlines.append(hds)
                        articles.append([hstyle] + content)
            
        cnt += 1
        if cnt % 1000 == 0:
            print("{0} processed {1} articles".format(fname,cnt))
            break
    print("finish processing file {0} with articles {1}".format(fname,len(headlines)))
    return (headlines,articles)

def write_callback(data):
    headlines, articles = data
    print("write {0} articles to file".format(len(headlines)))
    for title,content in zip(headlines,articles):
        source_file.write(" ".join(content).lower() + '\n')
        target_file.write(" ".join(title).lower() + '\n')

    
if __name__ == "__main__":
    words = set()
            
    # 添加词表
    fashion_words = pickle.load(open("/data/xueyou/fashion/sku/fashion_words.0302.pkl",'rb'))
    for word in fashion_words:
        if word:
            words.add(word.lower())

    def add_words_from_txt(fnames):
        for fname in fnames:
            for line in open(fname):
                word = line.strip().lower()
                if word:
                    words.add(word)

    add_words_from_txt([
        "/data/xueyou/fashion/words/category_words_0130.txt",
        "/data/xueyou/fashion/words/topic_words_0207.txt",
        "/data/xueyou/fashion/sku/words/fashion.words.1123.txt"])

    print("add {0} words to jieba dict".format(len(words)))
    for w in words:
        jieba.add_word(w)
    
    source_file_name = "/data/xueyou/textsum/headline/source.txt"
    source_file = open(source_file_name,'w')
    
    target_file_name = "/data/xueyou/textsum/headline/target.txt"
    target_file = open(target_file_name,'w')
    
    pool = mp.Pool(processes = 20)

    fashion_web_articles = pickle.load(open('/data/xueyou/fashion/sku/articles/fashion_web_articles.1122.pkl','rb'))
    print("start to process fashion web articles")
    # test
    #hds,cts = worker_fashion_web("fashion_web_articles" ,fashion_web_articles)
    pool.apply_async(worker_fashion_web,("fashion_web_articles" ,fashion_web_articles),callback=write_callback)
    
    print("start to process weixin high click articles")
    # test
    #hds,cts = worker_file("/data/xueyou/textsum/data/weixin_click8k_like80_len150_2017/000000_0",6,1,5,2,False)
    for filename in glob.glob('/data/xueyou/textsum/data/weixin_click8k_like80_len150_2017/*_0'):
        pool.apply_async(worker_file,(filename,6,1,5,2,False),callback=write_callback)

    print("start to process tmall articles")
    #hds,cts = worker_file("/data/xueyou/fashion/data/tmall_articles/000069_0",7,3,2,1,True)
    
    for filename in glob.glob("/data/xueyou/fashion/data/tmall_articles/*_0"):
        pool.apply_async(worker_file,(filename,7,3,2,1,True),callback=write_callback)

    print("start to process taobao headline articles")
    #hds,cts = worker_file("/data/xueyou/fashion/data/taobao_headline_0326/000000_0",15,1,3,None,False)
    for filename in glob.glob("/data/xueyou/fashion/data/taobao_headline_0326/*_0"):
        pool.apply_async(worker_file,(filename,14,1,4,None,False),callback=write_callback)
        
    pool.close()
    pool.join()
    source_file.close()
    target_file.close()
    

    print("shuffle training data")
    source_file_name,target_file_name = shuffle_big_files([source_file_name,target_file_name])

    print("process file to get vocab and statistics of data")
    title_count = Counter()
    content_count = Counter()
    word_count = Counter()
    style_count = Counter()
    
    def get_st():
        title_set = set()
        def process_line(line):
            line = line.replace(u'\u200b', '').strip()
            line = re.sub('\s+',' ',line)
            return line
    
        with open(source_file_name) as sf:
            with open(target_file_name) as tf:
                s,t = sf.readline(),tf.readline()
                cnt = 0
                while s and t:
                    s,t = process_line(s),process_line(t)
                    if s not in title_set:
                        title_set.add(s)
                        yield cnt,s,t
                    cnt += 1
                    s,t = sf.readline(),tf.readline()
        print("Done read data with size {0}".format(cnt))
    
    train_sf = open("/data/xueyou/textsum/headline/train.source",'w')
    train_tf = open("/data/xueyou/textsum/headline/train.target",'w')
    # 10000
    dev_cnt = 10000
    dev_sf = open("/data/xueyou/textsum/headline/dev.source",'w')
    dev_tf = open("/data/xueyou/textsum/headline/dev.target",'w')
    # 10000
    test_cnt = 10000
    test_sf = open("/data/xueyou/textsum/headline/test.source",'w')
    test_tf = open("/data/xueyou/textsum/headline/test.target",'w')
    
    
    for i,rcontent,rtitle in tqdm(get_st()):
        if dev_cnt != 0:
            dev_sf.write(rcontent + '\n')
            dev_tf.write(rtitle + '\n')
            dev_cnt -= 1
            continue
        
        if test_cnt != 0:
            test_sf.write(rcontent + '\n')
            test_tf.write(rtitle + '\n')
            test_cnt -= 1
            continue
            
        train_sf.write(rcontent + '\n')
        train_tf.write(rtitle + '\n')
        
        title = rtitle.split(' ')
        content = rcontent.split(' ')
        # remove style
        style_count[content[0]] += 1
        content = content[1:]
        
        title_count[len(title)] += 1
        content_count[len(content)] += 1
        word_count.update(content)
        word_count.update(title)
    
    pickle.dump({"title_count":title_count,"content_count":content_count,"word_count":word_count,"style_count":style_count},open("/data/xueyou/textsum/data/count.pkl",'wb'))
    
    train_sf.close()
    train_tf.close()
    dev_sf.close()
    dev_tf.close()
    test_sf.close()
    test_tf.close()
    
    word2id = {'<unk>':0,'<s>':1,'</s>':2}
    # add 4 style
    # 疑问，感叹，陈述，省略
    styles = ['<style-question>','<style-exclamation>','<style-declarative>','<style-ellipsis>']
    for style in styles:
        word2id[style] = len(word2id)
        
        
    print("Style count:")
    for s,c in style_count.most_common():
        print(s,c)
        
    min_count = 10
    print("get vocab")
    for w,c in word_count.most_common():
        if len(w) >= 20:
            continue
        if len(word2id) == 250000:
            break
        if c < min_count:
            break
        word2id[w] = len(word2id)
    
    print("write vocab to file with size:",len(word2id))
    with open("/data/xueyou/textsum/headline/vocab.250000.txt",'w') as f:
        words = sorted(word2id,key=lambda x:word2id[x])
        for w in words:
            f.write(w)
            f.write('\n')
        
    tc = 0
    cnt = 0
    min_l = 10000
    max_l = -1000
    for l,c in title_count.most_common():
        tc += l*c
        cnt += c
        
        min_l = min(l,min_l)
        max_l = max(l,max_l)
    print('average length of title:',tc/cnt)
    print('min length of title:',min_l)
    print('max length of title:',max_l)
    
    
    tmp = 0
    for l,c in sorted(title_count.most_common(),key=lambda item:item[0]):
        tmp += c
        if tmp >= cnt //2:
            print("median length of title:",l)
            break
            
    tc = 0
    cnt = 0
    min_l = 10000
    max_l = -1000
    for l,c in content_count.most_common():
        tc += l*c
        cnt += c
        min_l = min(l,min_l)
        max_l = max(l,max_l)
    print('average length of content:',tc/cnt)
    print('min length of content:',min_l)
    print('max length of content:',max_l)
    
    tmp = 0
    for l,c in sorted(content_count.most_common(),key=lambda item:item[0]):
        tmp += c
        if tmp >= cnt //2:
            print("median length of content:",l)
            break
    print("All done")
    
