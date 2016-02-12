from subprocess import check_output
import json
import gzip

def espeak(words):
    return check_output(["espeak", "-q", "--ipa",
                        '-v', 'en-us',
                        words]).decode('utf-8')

coco = json.load(open('/home/gchrupala/repos/reimaginet/data/coco/dataset.json'))

with gzip.open("/home/gchrupala/repos/reimaginet/data/coco/dataset.ipa.jsonl.gz","w") as f:
    for img in coco['images']:
        for s in img['sentences']:
            ipa = espeak(s['raw'])
            f.write("{}\n".format(json.dumps({'sentid':s['sentid'], 'ipa':ipa})))



