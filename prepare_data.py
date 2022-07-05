import argparse
from functools import wraps
import json
import pickle
import time
from typing import Callable
import pandas as pd
from pathlib import Path
import spacy
from utils import add_special_tokens, makedir, get_tokenized_text, prepare_input_summarizer

CONFIG = dict(
	# Useful columns for fine-tuning.
	cols=["id", "summary", "text"]
)


def write_json(i,article, abstract, base_dir="summaries/tokenized/"):
	""" Saves a json file."""
	with open(Path(base_dir, f'{i}.json'), 'w+') as f:
		json.dump(
			dict(
				id=i,
				article=article,
				abstract=abstract
			), 
			f, ensure_ascii=False)

#Not used
def load_json_data(func:Callable) -> Callable:
	"""Load json data from disk

	:param func: callable function
	:type func: Callable
	"""
	@wraps(func)
	def inner(filepath, *args, **kwargs):
		data = pd.read_json(filepath, lines=True)[CONFIG["cols"]]
		return func(data, filepath, *args, **kwargs)
	return inner

# @load_json_data
def generate_only_valid_data(datapath, max_tokens, mode, tokenizer):
	""" Reads file, extract articles and summaries, tokenize them and save as json files
		Args:
			file_names: list, all the articles with total no of tokens less than 1024
			directory: string, directory where files in file_names is stored
	"""
	tokenizer = add_special_tokens(tokenizer=tokenizer)
	
	data = pd.read_json(datapath, lines=True)[CONFIG["cols"]]

	base_dir = Path(Path(datapath).parent.parent, f'tokenized_{max_tokens}', mode)

	# Create the directory, if it does not exist.
	makedir(base_dir)

	print("Execution Started...")
	train_ids = []
	file_id_map = {}
	i = 0

	# fix hardcoded props.
	for idx, text, summary in zip(data.id, data.text, data.summary):
		tok_text, tok_summary = get_tokenized_text(text, tokenizer), get_tokenized_text(summary, tokenizer)
		# Save one token for the separation token between the article and its summary
		if len(tok_text)>0 and len(tok_summary)>0 and (len(tok_text)+len(tok_summary))<=max_tokens-1:
			train_ids.append(i)
			write_json(i,tok_text,tok_summary,base_dir=base_dir)
			file_id_map[i] = idx
			i += 1
			if i%100==0:
				print(i, " files written")

	print("saving file_id_map...")
	with open(Path(Path(base_dir).parent, f"file_id_map_{mode}_{max_tokens}.pickle"), 'wb') as f:
		pickle.dump(file_id_map,f)
	print("file_id_map saved.")

def generate_data_all(datapath, max_tokens, mode, tokenizer, spacy_model='es_core_news_sm'):
	tokenizer = add_special_tokens(tokenizer=tokenizer)
	
	data = pd.read_json(datapath, lines=True)[CONFIG["cols"]]

	base_dir = Path(Path(datapath).parent.parent, f'tokenized_all_{max_tokens}', mode)

	makedir(base_dir)

	spacy_model = spacy.load(spacy_model)
	
	i = 0
	file_id_map = {}


	for idx, text, summary in zip(data.id, data.text, data.summary):
		tok_summary = get_tokenized_text(summary, tokenizer)
		tok_text = prepare_input_summarizer(text,
											tokenizer,
											max_input=max_tokens,
											gpt2_summary_length=len(tok_summary),
											as_tokens=True,
											spacy_model=spacy_model)[0]
		if len(tok_text)>0 and len(tok_summary)>0 and (len(tok_text)+len(tok_summary))>max_tokens-1:
			print(f'Ilegal input sample: idx, {idx}; i {i}')
		else:
			write_json(i,tok_text,tok_summary,base_dir=base_dir)
			file_id_map[i] = idx
			i += 1
		if i%100==0:
			print(i, " files written")

	print("saving file_id_map...")
	with open(Path(Path(base_dir).parent, f"file_id_map_{mode}_{max_tokens}.pickle"), 'wb') as f:
		pickle.dump(file_id_map,f)
	print("file_id_map saved.")


def filter_data(datapath, max_tokens, mode, tokenizer):
	""" Reads file, extract articles and summaries, tokenize them and save as json files
		Args:
			file_names: list, all the articles with total no of tokens less than 1024
			directory: string, directory where files in file_names is stored
	"""
	tokenizer = add_special_tokens(tokenizer=tokenizer)
	
	data = pd.read_json(datapath, lines=True)[CONFIG["cols"]]

	base_dir = Path(Path(datapath).parent, f"filtered_{max_tokens}")

	# Create the directory, if it does not exist.
	makedir(base_dir)

	print("Execution Started...")
	i = 0

	# fix hardcoded props.
	with open(Path(base_dir, f"{mode}_{max_tokens}.jsonl"), 'w+', encoding="utf-8") as f:
		for idx, text, summary in zip(data.id, data.text, data.summary):
			tok_text, tok_summary = get_tokenized_text(text, tokenizer), get_tokenized_text(summary, tokenizer)
			# Save one token for the separation token between the article and its summary
			if len(tok_text)>0 and len(tok_summary)>0 and (len(tok_text)+len(tok_summary))<=max_tokens-1:
				json.dump(
					dict(
						id=idx,
						text=text,
						summary=summary
					), 
					f, ensure_ascii=False)
				f.write(f'\n')
				i += 1
				if i%100==0:
					print(i, " files written")

# How to execute (example):
# Train
# python prepare_data.py summaries/all/train.jsonl train -m 512 -t "PlanTL-GOB-ES/gpt2-base-bne"
# Validation
# python prepare_data.py summaries/all/val.jsonl val -m 512 -t "PlanTL-GOB-ES/gpt2-base-bne"
# Test
# python prepare_data.py summaries/all/test.jsonl test -m 512 -t "PlanTL-GOB-ES/gpt2-base-bne"

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("filepath", help="Path to the file containing the collection of summaries.")
	parser.add_argument("mode", choices=["train","val","test"], help="Type of data used.")
	parser.add_argument("--only_valid", action="store_true", help="Include only samples satisfying the restriction for the number of tokens of the article and the summary to be equal or smaller than the maximum input lenght allowed by the model.")
	parser.add_argument("-m", "--max_tokens", type=int, default=512, help="Maximum length for input samples. Note that a sample is an article along with its summary.")
	parser.add_argument("-f", "--filter_mode", action="store_true", help="Filter mode")
	parser.add_argument("-t", "--tokenizer", default="PlanTL-GOB-ES/gpt2-base-bne", help="Name of a pretrained tokenizer.")
	args = parser.parse_args()
	start = time.time()

	if args.filter_mode:
		target_f = filter_data
	elif args.only_valid:
		target_f = generate_only_valid_data
	else:
		target_f = generate_data_all

	target_f(args.filepath, args.max_tokens, args.mode, args.tokenizer)

	print("total_time_taken: ", (time.time()-start)/60, " minutes")