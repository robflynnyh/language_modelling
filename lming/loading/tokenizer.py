import sentencepiece as spm

def load_tokenizer(model_path:str = '/exp/exp4/acp21rjf/language_modelling/artifacts/tokenizer/tokenizer.model') -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=model_path)