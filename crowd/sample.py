# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from pycocotools.coco import COCO
import numpy
import random
import csv


def main():
    random.seed(123)

    dataDir='/home/gchrupala/repos/coco'
    dataType='val2014'
    cap = COCO('%s/annotations/captions_%s.json'%(dataDir,dataType))
    coco = COCO('%s/annotations/instances_%s.json'%(dataDir,dataType))
    imgCat = {}
    for cat,imgs in coco.catToImgs.items():
        for img in imgs:
            if img in imgCat:
                imgCat[img].add(cat)
            else:
                imgCat[img]=set([cat])

    with open('hard2.csv','w') as file:
        writer = csv.writer(file)
        writer.writerow(["desc", "url_1", "url_2", "url_3", "url_4" ])
        imgIds = random.sample(coco.getImgIds(), 1000)
        for img in coco.loadImgs(imgIds):
            if img['id'] not in imgCat:
                continue
            cats = imgCat[img['id']]
            desc = random.sample(cap.imgToAnns[img['id']],1)[0]
            imgs = coco.loadImgs(random.sample(sum([ coco.getImgIds(catIds=[cat]) 
                                                     for cat in cats ],[]),3))
            urls = [ img['coco_url'] ] + [ img['coco_url'] for img in imgs ]
            random.shuffle(urls)
            writer.writerow([desc['caption']] + urls )


main()
