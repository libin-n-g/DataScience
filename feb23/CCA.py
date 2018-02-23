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
            english_words.extend(f.read().split())
        with open(i,"r") as e:
            french_words.extend(e.read().split())
    french_words = map(lambda x:x.lower(), french_words)
    english_words = map(lambda x:x.lower(), english_words)
    regex = re.compile(r'\d+(\.\d*)?')
    french_words = filter(lambda i: (not regex.search(i) or i=='.'),french_words)
    english_words = filter(lambda i: (not regex.search(i) or i=='.'),english_words)
    fr = Counter(french_words)
    eg = Counter(english_words)
    return (fr,eg)
def plot(f_dict, e_dict):
    print "test"
    y =  sorted(e_dict.values(), reverse = True)
    print y
    x = range(len(y))
    plt.plot(x, y)
    plt.xlabel('Rank')
    plt.ylabel('word count')
    plt.show()
    
s,q = make_dict(get_files())
l = len(s.values())
print l
f = 1 - (5000/float(l))
print f
sss = sorted(s.values(),reverse = True)
print sss
print sss[30000]
lower = sss[30000]
upper = sss[1000]
print sss[1000:30000]
print upper, lower
for k,v in s.iteritems():
    #print k,v
    if ((v >= lower) and (v <= upper)):
        print "found"
a = {k:v for k,v in s.iteritems() if v >= lower and v <= upper}
print a
print sorted(a.values())
plot(q, a)
#print sorted(s.values())
#print sorted(s.values())[-len(s.values())/4:]
#print s['.']

