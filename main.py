import Video_transcription
import Abstractive_summarizer
from nltk.tokenize import sent_tokenize, word_tokenize

out=[None]

def pre_load():
    Video_transcription.load_model()
    Abstractive_summarizer.load_model()


def complie(path, link, out_key):
    global out
    
    ## transcription
    transcription = ""
    if path is not None:
        transcription = Video_transcription.get_transcribe_from_path(path, return_list=False, sep=".")
    elif link is not None:
        transcription = Video_transcription.get_transcribe_from_youtube(link, return_list=False, sep=" ")
    
    ## words summarize
    words = word_tokenize(transcription)
    summarie = Abstractive_summarizer.chunk_summarized(words, window_size=150, overlap=30)
    
    
    out = {
        "Transcribe": transcription,
        "Summarize": summarie
    }
    return out[out_key]


def show_output(key):
    return out[key]
    