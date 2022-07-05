import datetime
from functools import wraps
import random
from typing import Any, Callable, Iterable, Union
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import pickle
import spacy
import json
from rouge import Rouge
import pandas as pd

def compute_rouge_score(test_summaries:Iterable, generated:Iterable, results_path:Union[str, Path], score_format="csv"):
    """Compute average ROUGE scores of `generated` summaries, using `test_summaries` as reference.

    Parameters
    ----------
    test_summaries : Iterable
        Reference summaries
    generated : Iterable
        Generated summaries
    results_path : Union[str, Path]
        Filepath to store ROUGE metrics
    score_format : str, optional
        Format to store assessment results, by default "csv"
    """
    rouge = Rouge()

    scores = rouge.get_scores(generated, test_summaries, avg=True)

    if score_format=="csv":
        pd.DataFrame(scores).to_csv(results_path)
    elif score_format=="json":
        with open(results_path, 'w+') as f:
            f.write(json.dumps(scores))

    print('ROUGE results stored in ', results_path)


def load_serialized_data(filename:str, return_dict_values:bool=False) -> Union[dict, Any]:
    """Utility to load serialized data (and other optional stored values)
    from disk using *pickle*.
    
    :param str filename: Filename of the file to be loaded.
    :param return_dict_values: If set to True, returns the values just the values
     of the dictionary containing all stored data, defaults to False.
    :type return_dict_values: bool, optional
    :return: Loaded data
    :rtype: Union[dict, Any]
    """
    # Load embeddings and other stored information from disk
    with open(Path(filename), "rb") as fIn:
        stored_data = pickle.load(fIn)
    return stored_data.values() if return_dict_values else stored_data

def store_serialized_data(data, out_filename, protocol:int=pickle.HIGHEST_PROTOCOL) -> None:
    """Utility to dump precomputed data to disk using *pickle*.

    Parameters
    ----------
    data : _type_
        Data to serialize
    out_filename : str, optional
        Path for the output file
    protocol : int, optional
        Protocol used for *pickle*, by default pickle.HIGHEST_PROTOCOL
    """
    # Create directory if it does not exist.
    makedir(out_filename, remove_filename=True)
    with open(out_filename, "wb") as fOut:
        pickle.dump(data, fOut, protocol=protocol)

def add_special_tokens(tokenizer="PlanTL-GOB-ES/gpt2-base-bne"):
	""" Returns GPT2 tokenizer after adding separator and padding tokens

    Parameters
    ----------
    tokenizer : str, optional
        Model id of a pretrained HuggingFace tokenizer hosted inside a model repo on 
        huggingface.co , by default "PlanTL-GOB-ES/gpt2-base-bne"

    Returns
    -------
        GPT2 tokenizer after adding separator and padding tokens
    """
	tokenizer = AutoTokenizer.from_pretrained(tokenizer)
	special_tokens = {'pad_token':'<|pad|>','sep_token':'<|sep|>'}
	num_add_toks = tokenizer.add_special_tokens(special_tokens)
	return tokenizer

def set_seed(seed:int, gpu_mode:bool):
    """Set initialization state of a pseudo-random number generator to grant reproducibility 
    of the experiments

    Parameters
    ----------
    seed : int
        Seed
    gpu_mode : bool
        Whether there are GPU's available
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu_mode > 0:
        torch.cuda.manual_seed_all(seed)

def format_time(elapsed:float) -> str:
    """Format time in seconds to hh:mm:ss time format.

    Parameters
    ----------
    elapsed : float
        Time in seconds

    Returns
    -------
    str
        Time formatted in hh:mm:ss
    """
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def makedir(path:Union[str,Path], remove_filename:bool=False, recursive:bool=True, exist_ok:bool=True)->None:
	"""Creates directory from path if not exists.

    Parameters
    ----------
    path : Union[str,Path]
        Path of the directory to be created.
    remove_filename : bool, optional
        If set to True, it attempts to remove the filename from the path, defaults
        to False
    recursive : bool, optional
        Creates directories recursively (i.e., create necessary subdirectories if 
        necessary), by default True
    exist_ok : bool, optional
        If set to False, it arises an error if `path` directory exists, by default True
    """
	# Ensure it is a PurePath
	path = Path(path)
	
	if remove_filename: path = path.parent

	path.mkdir(parents=recursive, exist_ok=exist_ok)

def get_tokenized_text(text:str, tokenizer, convert_ids_to_tokens=False) -> list:
    """Returns tokenized text using the tokenizer `tokenizer`

    Parameters
    ----------
    text : str
        Text to tokenize
    tokenizer : `PreTrainedTokenizer` or `PreTrainedTokenizerFast`
        Tokenizer
    convert_ids_to_tokens : bool, optional
        Whether to convert ids to tokens, by default False

    Returns
    -------
    list
        Tokenized text
    """
    ids = tokenizer.encode(text)

    # Return list of tokens in `text`.
    return ids if not convert_ids_to_tokens else tokenizer.convert_ids_to_tokens(ids)

#python -m spacy download es_core_news_sm
def split_text_into_sentences_spacy(text, spacy_model='es_core_news_sm'):
    """Splits text into sentences using the Spacy library.

    Parameters
    ----------
    text : str
        Text to be splitted into sentences.
    spacy_model : str or SpaCy pretrained model object, optional
        SpaCy pretrained model used to split text into sentences or pretrained
        model identifier, defaults to 'es_core_news_sm'.

    Returns
    -------
    list
        List of sentences in `text`.

    Notes
    ------
    SpaCy builds a syntactic tree for each sentence, a robust method that yields 
    more statistical information about the text than NLTK. It performs substancially
    better than NLTK when using not polished text.
    """
    if type(spacy_model) == str:
        print(f'Loading SpaCy pretrained model: {spacy_model}')
        spacy_model = spacy.load(spacy_model)
    
    # sent_tok = spacy.load(spacy_model)
    return [i.text.strip() for i in spacy_model(text).sents]

def detokenize_input(func:Callable) -> Callable:
    """Detokenize text input, when required.

    Parameters
    ----------
    func : Callable
        Callable function

    Returns
    -------
    Callable
        Callable object with the appropiate parametrization 
    """
    @wraps(func)
    def inner(text, tokenizer, *args, **kwargs):
        # Attempt to detokenize text
        if type(text) != str:
            text = tokenizer.decode(text, skip_special_tokens=True).replace('</s>', '')
        return func(text, tokenizer, *args, **kwargs)
    return inner

@detokenize_input
def prepare_input_summarizer(text:str, tokenizer, max_input=512, gpt2_summary_length:int=None, as_tokens=False, **spacy_kwargs):
    """Prepare input to be handled by the Transformer model

    Parameters
    ----------
    text : str
        Raw, unprocessed text
    tokenizer : `PreTrainedTokenizer` or `PreTrainedTokenizerFast`
        Tokenizer object used to control for the number of tokens in `text`
    max_input : int, optional
        Maximum length, in terms of tokens, that the input can handle, by default 512
    gpt2_summary_length : int, optional
        Length, in terms of tokens, reserved to generate a summary when using decoder-only
        based summarizers (e.g., GPT-2), by default None. If you wish to use a different
        architecture (e.g., encoder-decoder), do NOT specify this argument.
    as_tokens : bool, optional
        Return the input as tokens (hence not as plain text).

    Returns
    -------
    (Union[str,torch.Tensor], bool)
        Prepared input to be handled by a Transformer and a flag indicating whether the text
        has been trimmed.
    """
    max_article_length = max_input - 1 # 1 token reserved for <|sep|>

    # Using decoder-only systems, both the summary and the article altogether should fit
    # into the model
    if gpt2_summary_length:
        max_article_length -= gpt2_summary_length

    # Whether the text has been trimmed.
    trimmed = False

    # Get tokenized text
    text_tok = get_tokenized_text(text, tokenizer)

    # If the length of the input is valid, then do nothing
    if len(text_tok) <= max_article_length:
        return (text, trimmed) if not as_tokens else (text_tok, trimmed)

    # If this code is executed, then the text input must be trimmed.
    trimmed = True

    # A non-exhaustive list of sentence separators
    SEPARATORS=frozenset([".", "!", "?", ";", ":"])

    # Split text into sentences using SpaCy
    sents = split_text_into_sentences_spacy(text, **spacy_kwargs)
    
    # Text_out will eventually hold the text valid for GPT-2/RoBERTa
    text_out = ""
    
    # Build new text from sentences until the number of tokens in the text exceeds
    # `max_article_length` in number of tokens.
    for sent in sents:
        # Add a space between sentences if they end in a punctuation symbol
        add_space = len(text_out) > 1 and text_out[-1] in SEPARATORS
        # Calculate the number of tokens of the resulting text after adding
        # a new sentence
        num_tok = len(get_tokenized_text(text_out + sent, tokenizer))
        # If a space must be added between sentences, then reserve a token to that end
        if add_space:
            num_tok += 1
        # Check whether `sent` can be included in the article. If not, stop.
        if num_tok > max_article_length:
            break
        # Otherwise, add the new sentence
        text_out += " " + sent if add_space else sent
    
    # Return text + trimmed status
    return (text_out, trimmed) if not as_tokens else (get_tokenized_text(text_out, tokenizer), trimmed)