import argparse
import datasets
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, EncoderDecoderModel
from pathlib import Path
from utils import makedir
from rouge import Rouge
import json

# Generate the text without setting a decoding strategy
def generate_summary(batch, max_length=512):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    #outputs = roberta_shared.generate(input_ids, attention_mask=attention_mask)
    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

# Generate a text using beams search
def generate_summary_beam_search(batch, max_length=512, num_beams=4):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(input_ids, attention_mask=attention_mask,
                                  num_beams=num_beams,
                                  num_return_sequences = 1
    )

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

# Generate a text using beams search
def generate_summary_top_k_top_p(batch, max_length=512, top_k=10, top_p=0.5):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(input_ids, attention_mask=attention_mask,
                                  num_return_sequences = 1,
                                  do_sample=True,
                                  top_k=top_k, 
                                  top_p=top_p,
    )

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

def generate_summary_combined(batch, max_length=512, repetition_penalty=2.5, num_beams=4, top_k=10, top_p=0.5):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(input_ids, attention_mask=attention_mask,
                                  num_return_sequences = 1,
                                  do_sample=True,
                                  repetition_penalty=repetition_penalty,
                                  num_beams=num_beams,
                                #   top_k=top_k, 
                                #   top_p=top_p,
    )

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

def compute_rouge_score(test_summaries, generated, gen_summaries_path, score_format="csv"):
    rouge = Rouge()

    scores = rouge.get_scores(generated, test_summaries, avg=True)

    if score_format=="csv":
        pd.DataFrame(scores).to_csv(gen_summaries_path)
    elif score_format=="json":
        with open(gen_summaries_path, 'w+') as f:
            f.write(json.dumps(scores))

def attrs_to_str(args, add:str=None, change=False) -> str:
    """Yield a string-like representation of the training attributes to fine-tune the model,
    serving as a straightforward and effective fashion to uniquely identify the model. 
    
    Parameters
    ----------
    add : str, optional
        Substring to append at the end of the string model descriptor, by default None

    Returns
    -------
    str
        Textual representation of the training parameters utilized for fine-tuning
    """
    enc_dec_type = "SHARED" if args.tie_weights else "R2R"
    data = Path(args.train_data_dir).name
    checkpoint_name = f"roberta-{args.model}-bne"

    train_epochs = args.num_train_epochs if change else args.num_train_epochs
    fstring = f"{checkpoint_name}_{enc_dec_type}_{data}_epochs_{train_epochs}_batch_{args.batch_size}_gradsteps_{args.gradient_accumulation_steps}_min{args.train_summary_min_length}_max{args.train_summary_max_length}"
    return fstring if not add else f'{fstring}_{add}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, help="Parent directory containing the training dataset on which the model has been trained.")
    parser.add_argument("--test_data_dir", type=str, help="Parent directory containing the test dataset.")
    parser.add_argument("--model", default='base', choices=["base", "large"], help="Type of BSC RoBERTa architecture")
    parser.add_argument("--tie_weights", action="store_true", help="Tie encoder decoder weights")
    parser.add_argument("--batch_size", type=int, required=True, help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps",default=32, type=int, required=False, help="Accumulated gradients run integer K small batches of size N before doing a backward pass.")
    parser.add_argument("--summary_min_length", type=int, required=False, help="Minimum length of the decoder output.")
    parser.add_argument("--summary_max_length", type=int, required=False, help="Maximum length of the decoder output.")
    parser.add_argument("--train_summary_min_length",default=56, type=int, required=False, help="Minimum length of the decoder output.")
    parser.add_argument("--train_summary_max_length",default=128, type=int, required=False, help="Maximum length of the decoder output.")
    parser.add_argument("--temperature",default=1.0, type=float, required=False, help="control the randomness of predictions by scaling the logits before applying softmax")
    parser.add_argument("--top_k",default=50, type=int, required=False, help="keep only top k tokens with highest probability (top-k filtering)")
    parser.add_argument("--top_p",default=0.9, type=float, required=False, help="keep the top tokens with cumulative probability >= top_p (nucleus filtering)")
    parser.add_argument("--num_beams",default=4, type=int, required=False, help="Number of beams in Beam search")
    parser.add_argument("-mo", "--model_dir",default='./models_encdec', type=str, required=True, help="Directory to save the trained model (and intermediate checkpoints)")
    parser.add_argument("--checkpoint_at_step", type=int, required=False, help="Load a checkpoit at a specific training step")
    parser.add_argument("--seed",default=42, type=int, required=False, help="Initialization state of a pseudo-random number generator to grant reproducibility of the experiments")
    args = parser.parse_args()

    test_data_dir = Path(args.test_data_dir)
    #Set the path to the data folder, datafile and output folder and files
    test_filepath = Path(args.test_data_dir, "test.jsonl")

    #TODO: delete!!!
    test_dataname = test_data_dir.name
    test_summaries_filename = Path(test_data_dir.parent, "test_summaries", f"test_summaries_{test_dataname}.txt")
    with open(test_summaries_filename, 'r', encoding='utf-8') as f:
        test_summaries = [line.replace('\n', '') for line in f.readlines()]


    model_folder = Path(args.model_dir, attrs_to_str(args, change=False))
    if args.checkpoint_at_step is not None:
        model_folder = Path(model_folder, f"checkpoint-{args.checkpoint_at_step}")

    checkpoint_name = f"PlanTL-GOB-ES/roberta-{args.model}-bne"

    config = dict(
        # Useful columns for fine-tuning.
        cols=["text", "summary"]
    )

    # load rouge for validation
    rouge = datasets.load_metric("rouge")

    test_df = pd.read_json(test_filepath, lines=True)[config["cols"]]
    test_dataset = Dataset.from_pandas(test_df)

    encoder_max_length = 512
    decoder_max_length = args.summary_max_length
    decoder_min_length = args.summary_min_length


    test_dataname = test_data_dir.name

    #Load the Tokenizer and the fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, max_len=encoder_max_length)
    model = EncoderDecoderModel.from_pretrained(model_folder)

    model.to("cuda")

    results = test_dataset.map(generate_summary_combined, batched=True, batch_size=args.batch_size, remove_columns=["text"], fn_kwargs=dict(max_length=encoder_max_length, num_beams=args.num_beams, top_k=args.top_k, top_p=args.top_p))
    pred_str_comb = results["pred"]
    #results = test_data.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["description"])
    # Generate predictions using beam search
    results = test_dataset.map(generate_summary_beam_search, batched=True, batch_size=args.batch_size, remove_columns=["text"], fn_kwargs=dict(max_length=encoder_max_length, num_beams=args.num_beams))
    pred_str_bs = results["pred"]
    # Generate predictions using top-k sampling
    results = test_dataset.map(generate_summary_top_k_top_p, batched=True, batch_size=args.batch_size, remove_columns=["text"], fn_kwargs=dict(max_length=encoder_max_length, top_k=args.top_k, top_p=args.top_p))
    pred_str_topk_top_p = results["pred"]

    results = test_dataset.map(generate_summary, batched=True, batch_size=args.batch_size, remove_columns=["text"], fn_kwargs=dict(max_length=encoder_max_length))
    pred_str = results["pred"]

    #label_str = results["Summary"]

    """Now, we can see some results from our trained model to check its performance on the task."""

    #Show an example
    print("Article: ",test_df['text'][1])
    print(" Summary using BS: ", pred_str_bs[1])
    print(" Summary using Top-K Sampling: ", pred_str_topk_top_p[1])

    #Show an example
    print("Article: ",test_df['text'][5])
    print(" Summary using BS: ", pred_str_bs[5])
    print(" Summary using Top-K Sampling: ", pred_str_topk_top_p[5])


    """Save the predictions to a file:"""
    test_dataname = test_data_dir.name
    gen_summaries_dir = Path(test_data_dir.parent, 'inference_encdec')
    makedir(gen_summaries_dir)

    test_summaries_filename = Path(test_data_dir.parent, "test_summaries", f"test_summaries_{test_dataname}.txt")
    with open(test_summaries_filename, 'r', encoding='utf-8') as f:
        test_summaries = [line.replace('\n', '') for line in f.readlines()]

    iminmax = f"_imin{decoder_min_length}_imax{decoder_max_length}" if args.summary_max_length != None or args.summary_min_length != None else ""

    output_combined=Path(gen_summaries_dir, f"gen_sum_{test_dataname}_{attrs_to_str(args)}_BEAM_{args.num_beams}_rep-penalty{2.5}{iminmax}.txt")
    rouge_combined = Path(gen_summaries_dir, f"rouge_{test_dataname}_{attrs_to_str(args)}_BEAM_{args.num_beams}_rep-penalty{2.5}{iminmax}.csv")
    with open(output_combined, 'w+', encoding='utf8') as f:
        for summary in pred_str_comb:
            f.write(summary.replace('""', '"') +'\n')

    compute_rouge_score(test_summaries, pred_str_comb, rouge_combined)


    output_beam=Path(gen_summaries_dir, f"gen_sum_{test_dataname}_{attrs_to_str(args)}_BEAM_{args.num_beams}{iminmax}.txt")
    rouge_beam_path = Path(gen_summaries_dir, f"rouge_{test_dataname}_{attrs_to_str(args)}_BEAM_{args.num_beams}{iminmax}.csv")
    with open(output_beam, 'w+', encoding='utf8') as f:
        for summary in pred_str_bs:
            f.write(summary.replace('""', '"') +'\n')

    compute_rouge_score(test_summaries, pred_str_bs, rouge_beam_path)


    output_topk_topp=Path(gen_summaries_dir, f"gen_sum_{test_dataname}_{attrs_to_str(args)}_topk_{args.top_k}_topp_{args.top_p}{iminmax}.txt")
    rouge_topk_topp_path = Path(gen_summaries_dir, f"rouge_{test_dataname}_{attrs_to_str(args)}_topk_{args.top_k}_topp_{args.top_p}{iminmax}.csv")
    with open(output_topk_topp, 'w+', encoding='utf8') as f:
        for summary in pred_str_topk_top_p:
            f.write(summary.replace('""', '"') +'\n')

    compute_rouge_score(test_summaries, pred_str_topk_top_p, rouge_topk_topp_path)

    output_asis=Path(gen_summaries_dir, f"gen_sum_{test_dataname}_{attrs_to_str(args)}_ASIS{iminmax}.txt")
    rouge_asis_path = Path(gen_summaries_dir, f"rouge_{test_dataname}_{attrs_to_str(args)}_ASIS{iminmax}.csv")
    with open(output_asis, 'w+', encoding='utf8') as f:
        for summary in pred_str:
            f.write(summary.replace('""', '"') +'\n')

    compute_rouge_score(test_summaries, pred_str, rouge_asis_path)