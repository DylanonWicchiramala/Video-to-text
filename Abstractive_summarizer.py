from torch import cuda
from config import config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from os import path
import util
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

DEVICE = config['compute_device']
MODEL_SOURCES = config['abstractive_summarizer']['model_sources']


def load_model():
    global tokenizer, model
    
    model_path = util.load_model_repo(MODEL_SOURCES)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # tokenizer = tokenizer.to(DEVICE)
    model = model.to(DEVICE)
    
    return tokenizer, model


@util.make_iterable
def sumerized(text:str, max_new_tokens=1000, do_sample=False) -> str:
    """get extractive summaries from text
    parameter:
        text: str
        max_new_tokens: int; the maximum number of tokens output from summerize model.
        do_sample: bool; make output random, if True.
    """
    input = tokenizer(text, max_length=None, return_tensors='pt', truncation=False).input_ids.to(DEVICE)
    tensor_out = model.generate(input, max_new_tokens=max_new_tokens, do_sample=do_sample)
    output = tokenizer.decode(tensor_out[0], skip_special_tokens=True)
    return output


def chunk_summarized(text_in_chk, window_size=3, overlap=0, max_new_tokens=1000, do_sample=False):
    """Get extractive summaries from text
    
    Parameters:
        text_in_chk (list): list of text splited in chunks.
        window_size (int): Chunk size, e.g., 2 means 2 sentences per chunk.
        overlap (int): Number of sentences overlapping between chunks.
        max_new_tokens (int): The maximum number of tokens in the output summary.
        do_sample (bool): Make output random if True.
    
    Returns:
        str: The extractive summary.
    """
    
    # Initialize an empty list to store the summary chunks
    summary_chunks = []
    
    # Iterate over the sentences with the specified window size and overlap
    for i in range(0, len(text_in_chk), window_size - overlap):
        chunk = text_in_chk[i:i + window_size]
        
        # Combine the chunk into a single string
        chunk_text = ' '.join(chunk)
              
        # Perform summarization on the chunk using a function like 'sumerized'
        # Note: You should replace 'sumerized' with your actual summarization function
        summarized_chunk = sumerized(chunk_text)
        
        # Append the summarized chunk to the list
        summary_chunks.append(summarized_chunk)
    
    # Combine the summary chunks into a single summary
    final_summary = ' '.join(summary_chunks)
    
    return final_summary


if  __name__ == '__main__':
    with open("sample\starfields.txt", 'r') as file:
        # Read the entire contents of the file into a string
        txt = file.read()
        

    words = word_tokenize(txt)

    # print(words)
    print(chunk_summarized(words, window_size=32, overlap=4))
