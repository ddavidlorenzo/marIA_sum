import argparse
import datasets
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, EncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from pathlib import Path
from utils import makedir, store_serialized_data



# load rouge for validation
rouge = datasets.load_metric("rouge")

def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # We have to make sure that the PAD token is ignored by the loss function
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    print(f'Reference string: {label_str[:10]}')
    print(f'Predicted string: {pred_str[:10]}')
    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }
    
def attrs_to_str(args, add:str=None) -> str:
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
    data = Path(args.data_dir).name
    checkpoint_name = f"roberta-{args.model}-bne"
    # add = "length_penalty_1_beam_1"
    fstring = f"{checkpoint_name}_{enc_dec_type}_{data}_epochs_{args.num_train_epochs}_batch_{args.batch_size}_gradsteps_{args.gradient_accumulation_steps}_min{args.summary_min_length}_max{args.summary_max_length}"
    return fstring if not add else f'{fstring}_{add}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Parent directory containing at least the training and validation datasets to fine tune the model. The data should be formatted in such way that it can be processed by a `GPT2SumDataset` object. Refer to the `prepare_data.py` script for further information")
    parser.add_argument("--model", default='base', choices=["base", "large"], help="Type of BSC RoBERTa architecture")
    parser.add_argument("--tie_weights", action="store_true", help="Tie encoder decoder weights")
    parser.add_argument("--batch_size", type=int, required=True, help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps",default=32, type=int, required=False, help="Accumulated gradients run integer K small batches of size N before doing a backward pass.")
    parser.add_argument("--summary_min_length",default=56, type=int, required=False, help="Minimum length of the decoder output.")
    parser.add_argument("--summary_max_length",default=128, type=int, required=False, help="Maximum length of the decoder output.")
    parser.add_argument("--lr",default=5e-5, type=float, required=False, help="Initial learning rate")
    parser.add_argument("-mo", "--model_dir",default='./models_encdec', type=str, required=True, help="Directory to save the trained model (and intermediate checkpoints)")
    parser.add_argument("-o", "--output_dir",default='./output_encdec', type=str, required=True, help="Directory to save the trained model and the evaluation results")
    parser.add_argument("--seed",default=42, type=int, required=False, help="Initialization state of a pseudo-random number generator to grant reproducibility of the experiments")
    args = parser.parse_args()

    #Set the path to the data folder, datafile and output folder and files
    data_dir = Path(args.data_dir)
    train_filepath = Path(data_dir, "train.jsonl")
    val_filepath = Path(data_dir, "val.jsonl")

    model_folder = Path(args.model_dir, attrs_to_str(args))
    makedir(model_folder)
    output_folder = Path(args.output_dir, attrs_to_str(args))
    makedir(output_folder)


    checkpoint_name = f"PlanTL-GOB-ES/roberta-{args.model}-bne"

    config = dict(
        # Useful columns for fine-tuning.
        cols=["text", "summary"]
    )

    # set decoding params   
    enc_dec_config = dict(
        early_stopping = True,
        no_repeat_ngram_size = 2,
        length_penalty = 1.5, # if 1 => no penalty
        num_beams = 4 # if 1 => no beam search
    )
    encoder_max_length = 512           # Max length for product description
    decoder_max_length = args.summary_max_length         # Max length for product names
    decoder_min_length = args.summary_min_length

    train_df = pd.read_json(train_filepath, lines=True)[config["cols"]]
    val_df = pd.read_json(val_filepath, lines=True)[config["cols"]]

    train_dataset=Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Load the trained tokenizer on our specific language
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, max_len=encoder_max_length)

    # Setting the BOS and EOS token
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    # Preprocessing the training data
    train_data = train_dataset.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=args.batch_size, 
        remove_columns=["text", "summary"]
    )
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"],
    )
    # Preprocessing the validation data
    val_data = val_dataset.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=args.batch_size, 
        remove_columns=["text", "summary"]
    )
    val_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"],
    )

    roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained(
        checkpoint_name, 
        checkpoint_name, 
        tie_encoder_decoder=args.tie_weights
        )
    
    # Show the vocab size to check it has been loaded
    print('Vocab Size: ',roberta_shared.config.encoder.vocab_size)

    # set special tokens
    roberta_shared.config.decoder_start_token_id = tokenizer.bos_token_id                                             
    roberta_shared.config.eos_token_id = tokenizer.eos_token_id
    roberta_shared.config.pad_token_id = tokenizer.pad_token_id

    # set decoding params   
    roberta_shared.config.max_length = decoder_max_length
    roberta_shared.config.min_length = decoder_min_length
    roberta_shared.config.early_stopping = enc_dec_config["early_stopping"]
    roberta_shared.config.no_repeat_ngram_size = enc_dec_config["no_repeat_ngram_size"]
    roberta_shared.config.length_penalty = enc_dec_config["length_penalty"]
    roberta_shared.config.num_beams = enc_dec_config["num_beams"]
    roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_folder,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        evaluation_strategy="steps",
        do_train=True,
        do_eval=True,
        logging_steps=100,  
        save_steps=200,
        eval_steps= 200,
        warmup_steps=200,
        num_train_epochs = args.num_train_epochs,
        overwrite_output_dir=True,
        save_total_limit=5,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=True, 
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        tokenizer=tokenizer,
        model=roberta_shared,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    # Fine-tune the model, training and evaluating on the train dataset
    history = trainer.train()

    # Save the encoder-decoder model just trained
    trainer.save_model(model_folder)