import numpy
import json

print "epoch r@1 R@5 r@10 rank"
for i in range(1,11):
    data = json.load(open("scores.{}.json".format(i)))
    print i, numpy.mean(data['recall']['1']), \
             numpy.mean(data['recall']['5']),\
             numpy.mean(data['recall']['10']),\
             numpy.median(data['ranks'])
    
    
