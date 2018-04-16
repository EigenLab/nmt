import pickle
import random
import statistics
import jieba
from collections import Counter
from tqdm import tqdm

print("loading data from pkl file")

data = pickle.load(open("/data/xueyou/fashion/data/sku_desc/sku_desc.0408.pkl",'rb'))

print("raw data size:",len(data))

print("filter data with empty sku or empty content")

data = [item for item in data if len(item['content'])>0 and len(item['sku']['sku_title'])>0]

print("filtered data size",len(data))

random.shuffle(data)

def stats(lens):
    print("min:",min(lens))
    print("max:",max(lens))
    mean = statistics.mean(lens)
    print("mean:",mean)
    print("median:",statistics.median(lens))
    stddev = statistics.stdev(lens)
    print("stddev:",stddev)
    return mean,stddev


def creat_training_data(data, word_level = False, max_vocab_size=0, prefix_dir = "/data/xueyou/data/sku_desc/"):
    sources = []
    targets = []
    for item in tqdm(data):
        sources.append(item['sku']['sku_title'].lower().replace(" ",'').strip())
        targets.append(item['content'].lower().replace(" ",'').strip())

    if word_level:
        print("load fashion words")
        for line in open("/data/xueyou/fashion/sku/words/fashion.words.1123.txt"):
            w = line.strip()
            if w:
                jieba.add_word(w)
        sources = [list(jieba.cut(src, HMM=False)) for src in tqdm(sources)]
        targets = [list(jieba.cut(tgt, HMM=False)) for tgt in tqdm(targets)]


    print("stats of sources")
    smean,sstddev = stats([len(x) for x in sources])

    print("stats of targets")
    tmean,tstddev = stats([len(x) for x in targets])

    # using 50k as dev and 50k as test
    training_sources = sources[:-100000]
    training_targets = targets[:-100000]
    dev_sources = sources[-100000:-50000]
    dev_targets = targets[-100000:-50000]
    test_sources = sources[-50000:]
    test_targets = targets[-50000:]

    word_cnt = Counter()

    for s,t in zip(training_sources,training_targets):
        word_cnt.update(s)
        word_cnt.update(t)

    min_cnt = 5

    word2id = {'<unk>':0,'<s>':1,'</s>':2}

    for w,c in word_cnt.most_common():
        if c < min_cnt:
            break
        if max_vocab_size and len(word2id) >= max_vocab_size:
            break
        word2id[w] = len(word2id)

    print("write vocab to file with size:",len(word2id))
    words = sorted(word2id,key=lambda x:word2id[x])
    with open(prefix_dir + "vocab.source",'w') as sf:
        with open(prefix_dir + "vocab.target",'w') as tf:
            for w in words:
                sf.write(w + '\n')
                tf.write(w + '\n')

    def write_data_to_file(sources,targets,prefix):
        with open(prefix_dir + prefix + ".source",'w') as f:
            for s in sources:
                f.write(" ".join(s) + '\n')
        with open(prefix_dir + prefix + ".target",'w') as f:
            for t in targets:
                f.write(" ".join(t) + '\n')
        
    print("write training data to file with size:",len(training_sources))
    write_data_to_file(training_sources,training_targets,"train") 
    write_data_to_file(dev_sources,dev_targets,"dev")    
    write_data_to_file(test_sources,test_targets,"test")    

#print("create training data with character")
#creat_training_data(data)
print("create training data with word")
creat_training_data(data,True,200000,"/data/xueyou/data/sku_desc/word_level/")    