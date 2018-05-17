import random
import statistics
from collections import Counter
import os

prefix_dir = "/data/xueyou/car/car_slot_data/char/"

data = []
with open("/data/xueyou/car/car_slot_data/data_char_level.txt") as f:
    for line in f:
        tokens = line.strip().split("\x01")
        if len(tokens)==2:
            data.append(tuple(tokens))

headlines = []
with open("/data/xueyou/car/car_slot_data/raw_headlines.txt") as f:
    for line in f:
        headlines.append(line.strip())

data_with_slots = []
with open('/data/xueyou/car/car_slot_data/content_slots.txt') as f:
    for i,line in enumerate(f):
        slots = line.strip().split("\x01")
        data_with_slots.append({"data":data[i],'slots':slots,'headline':headlines[i]})

    

print("data size",len(data_with_slots))
print("filter with length")
data_with_slots = [item for item in data_with_slots if len(item['data'][0].split()) >= 10]
print("data size",len(data_with_slots))

data = data_with_slots
random.shuffle(data)

train = data[:-1000]
dev = data[-1000:-500]
test = data[-500:]

def stats(lens):
    print("min:",min(lens))
    print("max:",max(lens))
    mean = statistics.mean(lens)
    print("mean:",mean)
    print("median:",statistics.median(lens))
    stddev = statistics.stdev(lens)
    print("stddev:",stddev)
    return mean,stddev


s_lens = []
t_lens = []

for item in train:
    s,t = item['data']
    s_lens.append(len(s.split()))
    t_lens.append(len(t.split()))

print("stats of source")
stats(s_lens)
print("stats of target")
stats(t_lens)

word_cnt = Counter()

for item in train:
    s,t = item['data']
    word_cnt.update(s.split())
    word_cnt.update(t.split())

min_cnt = 3
max_vocab_size = 100000

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

def write_data_to_file(out_data,prefix):
    with open(prefix_dir + prefix + ".source",'w') as sf:
        with open(prefix_dir + prefix + ".target",'w') as tf:
            with open(prefix_dir + prefix + '.slots', 'w') as slot_f:
                with open(prefix_dir + prefix + '.headline','w') as hf:
                    for item in out_data:
                        s,t = item['data']
                        sf.write(s + '\n')
                        tf.write(t + '\n')
                        slot_f.write("\x01".join(item['slots']) + '\n')
                        hf.write(item['headline'] + '\n')


print("write training data to file with size:",len(train))
write_data_to_file(train,"train") 
print("write dev data to file with size:",len(dev))
write_data_to_file(dev,"dev")    
print("write test data to file with size:",len(test))
write_data_to_file(test,"test")  
