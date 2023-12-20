import sentencepiece as spm
import os

def load_tokenizer(model_path:str = '../../artifacts/tokenizer/tokenizer.model') -> spm.SentencePieceProcessor:
    if model_path.startswith('../'):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
    return spm.SentencePieceProcessor(model_file=model_path)