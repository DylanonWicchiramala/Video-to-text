import os.path
from torch import cuda
from youtube_transcript_api import YouTubeTranscriptApi
import youtube_transcript_api
from faster_whisper import WhisperModel
import warnings
import re
from config import config


DEVICE = config['compute_device']
NUM_WORKERS = config['num_workers']
CPU_THREADS = config['cpu_threads']
MODEL_SIZE = config['whisper']['model_size']
COMPUTE_TYPE = config['whisper']['compute_type']
BEAM_SIZE = config['whisper']['beam_size']
VAD_FILTER = config['whisper']['vad_filter']


def load_model():
    """download Speech to text model faster-whisper from https://huggingface.co/guillaumekln
        parameters:
            model_size: large-v2, large-v1, medium, small, base, tiny, or tiny.en, base.en, etc.
    """
    global speech2text
    speech2text = WhisperModel(
        MODEL_SIZE, 
        device=DEVICE, compute_type=COMPUTE_TYPE, 
        num_workers = NUM_WORKERS,
        cpu_threads= CPU_THREADS,
        # download_root=os.path.abspath("models/faster-whisper"),
        )
    return speech2text


def get_transcribe_from_path(path, return_list=True, sep='\n'):
    """
    get transcribe from audio(or video) local file path using speed-to-text model (whisper model)
    parameter:
        path(str): path to audio file or video file.
        return_list(bool): if true its return transcription in list otherwise return in string sperate with newlines.
    return: list or string 
    """
    
    path = os.path.abspath(path)
    
    # transcribe the video (or audio).
    transcript, info = speech2text.transcribe(path, beam_size=BEAM_SIZE, vad_filter=VAD_FILTER)
    
    # transform dict to list of transcriptions
    texts = []
    for t in transcript:
        texts.append(t.text)
    
    # return transcriptions  
    if return_list:
        return texts
    else:
        return sep.join(texts)
    

def get_transcribe_from_youtube(link, return_list=True, sep='\n'):
    """
    get transcribe from youtube transcript api
    parameter:
        link(str): link to the video youtube. example: https://youtu.be/dQw4w9WgXcQ?si=ASXh0Axb5W8ecEve, https://www.youtube.com/watch?v=QlyfydvWrKY
        return_list(bool): if true its return transcription in list otherwise return in string sperate with newlines.
    return: list or string 
    """
    
    def get_video_id(link):
        """
        convert youtube link(str) to video id (str)
        """
        if bool(re.search(r'youtu.be/', link)):
            return link.split('/')[-1]
        if bool(re.search(r'watch\?v=', link)):
            return link.split('=')[-1]
        else:
            return link
        
    video_id = get_video_id(link)
    
    # get transcripe
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    
    except youtube_transcript_api._errors.TranscriptsDisabled:
        warnings.warn("Transcription are disabled for this YouTube video")
        transcript = ""
        
    # process transcribe, transform to list
    texts = []
    for t in transcript:
        if not bool(re.search(r'\[.*?\]', t['text'])):
            texts.append(t['text'])
        
    # return transcriptions
    if return_list:
        return texts
    else:
        return sep.join(texts)


# deprecated
# def get_transcribe_old(path=None, link=None, return_list=False):
    
#     if path is not None:
#         return get_transcribe_from_path(path, return_list)
    
#     if link is not None:
#         return get_transcribe_from_youtube(link, return_list)
    
    
def get_transcribe(path_or_link=None, return_list=False, sep="\n"):
    """ get the transcript form local audio (or video) path or YouTube link
    """
    
    
    if os.path.isfile(os.path.abspath(path_or_link)):
        return get_transcribe_from_path(path_or_link, return_list, sep)
    
    elif path_or_link is not None:
        return get_transcribe_from_youtube(path_or_link, return_list, sep)
    
    
def terminal_input():
    import time
    st = time.time()
    
    inp = input("Input local video path or YouTube link:\n")
    inp = "sample\Automatic Speech Recognition2.mp4" if inp=="" else inp
    t = get_transcribe(inp, return_list=False)
    
    print(t)
    print("compute time", time.time()-st)


if __name__=="__main__":
    terminal_input()