import os
import operator
import re
from collections import Counter
import matplotlib.pyplot as plt
def get_files(path="hansard.36/Release-2001.1a/sentence-pairs/senate/debates/development/training"):
    pairs = []
    for file in os.listdir(path):
        if file.endswith(".e"):
            pairs.append([os.path.join(path, file),os.path.join(path, file[:-1] + "f") ] )
    return pairs

def make_dict(file_pairs):
    french_words = [] 
    english_words = []
    for i,j in file_pairs:
        with open(j,"r") as f:
            english_words.extend(re.sub(r'[^\w]', ' ', f.read()).split())
        with open(i,"r") as e:
            french_words.extend(re.sub(r'[^\w]', ' ', e.read()).split())
    french_words = map(lambda x:x.lower(), french_words)
    english_words = map(lambda x:x.lower(), english_words)
    regex = re.compile(r'\d+(\.\d*)?')
    french_words = filter(lambda i: (not regex.search(i) or i=='.'),french_words)
    english_words = filter(lambda i: (not regex.search(i) or i=='.'),english_words)
    fr = Counter(french_words)
    eg = Counter(english_words)
    return (fr,eg)
def plot(e_dict):
    y =  sorted(e_dict.values(), reverse = True)
    x = range(len(y))
    plt.plot(x, y)
    plt.xlabel('Rank')
    plt.ylabel('word count')
    plt.show()
    

def filter_corpus(words, top = 0.1, bottom=0.8):
    wordcount = sorted(words.values())
    l = len(words.values())
    lower = wordcount[int(l*top)]
    upper = wordcount[int(l*bottom)]
    cutted = {k:v for k,v in words.iteritems() if v >= lower and v <= upper}
    plot(cutted)
    for k,v in cutted.iteritems():
        print k,v
    return cutted

s,q = make_dict(get_files())

s = filter_corpus(s,0.6, 0.98)
s = filter_corpus(q,0.6 , 0.98)

l = len(s.values())
print l


