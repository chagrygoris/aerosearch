import whisper
import logging
from typing import List
import time

logging.basicConfig(filename='/Users/chaykovsky/Desktop/projects_random/aerosearch/logs', level=logging.INFO)


model = whisper.load_model("large")

logging.info("Model loaded")


def transcribe_audio(au):
    start = time.time()
    res = model.transcribe(au)['text']
    end = time.time()
    logging.info(f"{au} transcribed: {res}, processing time {end - start}")
    return res


def transcribe_audios(audios:List[str]):
    '''
        str - filenames of .wav audios
    '''
    result = []


    for au in audios:

        start = time.time()
        
        result.append(model.transcribe(au)['text'])

        end = time.time()

        logging.info(f"{au} transcribed: {result[-1]}, processing time {end - start}")

    return result


if __name__ == '__main__':
    transcribe_audios([
        '../../src/tmp/chillers.wav'
    ])

