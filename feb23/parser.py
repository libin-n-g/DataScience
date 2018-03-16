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

def get_docs(filepath):
    files = []
    with open(filepath,"r") as f:
        docs = re.split("<TEXT>",f.read())
    for i in docs:
        s = i.split("</TEXT>")
        for j in s:
            if "</DOC>" in j or "<DOCNO>" in j:
                continue
            files.append(j)
    return files
def create_file(files):
    words = []
    with open("formated.txt", "w") as o:
        count = len(files)
        o.write(str(count))
        for f in files:
            words = (re.sub(r'[^\w]', ' ', f).split())
            english_words = map(lambda x:x.lower(), words)
            regex = re.compile(r'\d+(\.\d*)?')
            english_words = filter(lambda i: (not regex.search(i) or i=='.'),english_words)
            eg = Counter(english_words)
            words_sort = sorted(eg.keys())
            o.write("\n")
            for e in words_sort[0:100]:
                if len(e) < 3:
                    continue
                o.write(e + " ")
            
a = get_docs("ap.txt")
create_file(a)
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
