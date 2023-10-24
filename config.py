import os.path
from torch import cuda

config = {
    "models_path": os.path.abspath("models"),
    "compute_device": "auto", # options: auto, cpu, gpu
    "num_workers": 8,
    "cpu_threads": 0,

    "whisper": {
        "model_size": "base", # options: large-v2, large-v1, medium, small, base, tiny, or tiny.en, base.en, etc.
        "compute_type": "default", # options: default, int8, float16, etc https://opennmt.net/CTranslate2/quantization.html
        "beam_size": 5, # default: 5
        "vad_filter": True, # default: True
    },
    'abstractive_summarizer':{
        'model_sources': "https://huggingface.co/DylanonWic/mT5_summarize_th_en" 
        # https://huggingface.co/DylanonWic/mT5_summarize_th_en , 
        # https://huggingface.co/facebook/bart-base-xsum
    }
}

if config['compute_device'] == 'auto':
    config['compute_device'] = "cuda" if cuda.is_available() else "cpu"