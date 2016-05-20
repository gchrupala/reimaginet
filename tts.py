import json
import gzip
from gtts import gTTS
import StringIO
import base64

def speak(words):
    f = StringIO.StringIO()
    gTTS(text=words, lang='en-us').write_to_fp(f)
    return f.getvalue()

data = json.load(open('/home/gchrupala/repos/reimaginet/data/flickr8k/dataset.json'))

with gzip.open("/home/gchrupala/repos/reimaginet/data/flickr8k/dataset.speech.jsonl.gz","w") as f:
    for img in data['images']:
        for s in img['sentences']:
            ipa = speak(s['raw'])
            f.write("{}\n".format(json.dumps({'sentid':s['sentid'], 'speech':base64.b64encode(ipa)})))



