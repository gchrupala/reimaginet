import numpy
import imaginet.task as task
import imaginet.defn.audiovis_rhn as audiovis
import imaginet.tts as tts
import sys
import argparse
import logging
import os

def synthesize(text, path=None):
    return tts.decodemp3(tts.speak(text))

def activations(texts, model="/home/gchrupala/reimaginet/run-rhn-coco-9-resume/model.r.e9.zip",
                       audio_dir=None):
    """Return layer states and embeddings for sentences in texts, by synthesizing speech
       for each text, extracting MFCC features and applying a speech model to them.
    """
    logging.info("Loading model")
    model = task.load("/home/gchrupala/reimaginet/run-rhn-coco-9-resume/model.r.e9.zip")
    logging.info("Synthesizing speech")
    audios = [ synthesize(text) for text in texts]
    if audio_dir is not None:
        try:
            os.makedirs(audio_dir)
        except OSError:
            pass
        logging.info("Storing wav files")
        for i, audio in enumerate(audios):
            with open("{}/{}.wav".format(audio_dir, i), 'w') as out:
                out.write(audio)
    logging.info("Extracting MFCC features")
    mfccs  = [ tts.extract_mfcc(audio) for audio in audios]
    logging.info("Extracting layer states")
    states = audiovis.layer_states(model, mfccs)
    logging.info("Extracting sentence embeddings")
    embeddings = audiovis.encode_sentences(model, mfccs)
    return {'layer_states': states, 'embeddings': embeddings}


def main():
    logging.getLogger().setLevel('INFO')
    model_path="/home/gchrupala/reimaginet/run-rhn-coco-9-resume/model.r.e9.zip"
    parser = argparse.ArgumentParser()
    parser.add_argument('texts',
                            help='Path to file with texts')
    parser.add_argument('--model', default=model_path,
                            help='Path to file with model')
    parser.add_argument('--layer_states', default='states.npy',
                            help='Path to file where layer states will be stored')
    parser.add_argument('--embeddings', default='embeddings.npy',
                            help='Path to file where sentence embeddings will be stored')
    parser.add_argument('--audio_dir', default=None,
                            help='Path to directory where audio will be stored')
    args = parser.parse_args()
    texts = [ line.strip() for line in open(args.texts)]
    result = activations(texts, model=args.model, audio_dir=args.audio_dir)
    numpy.save(args.layer_states, result['layer_states'])
    numpy.save(args.embeddings, result['embeddings'])

if __name__=='__main__':
    main()
