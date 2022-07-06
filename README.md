# Applying Large Spanish Language Models to NLP Tasks

### Lorenzo Alfaro, David

#### ETSIINF, UPM

---

</br>

## Introduction

Abstractive summarization has experienced a surge of interest thanks to recent advancements on Transformer-based encoder-decoder models, with standout proposals like PEGASUS, that incorporates explicit pre-training objectives tailored for such a task, exhibiting unprecedented level of performance on natural language generation tasks. However, the humongous amount of data required and the massive computational cost attached to the pre-training of these architectures imposes a substantial burden on their availability in languages other than English, with the very exception of their multi-lingual homologous.

The recent large Spanish language models from the MarIA project, based on the RoBERTa and GPT-2 architectures, have shown promising results, pushing the state-of-the-art on multiple natural language understanding tasks. However, encoder- and decoder-only systems pose as an architecturally suboptimal approach to resolve sequence-to-sequence tasks. In this work, we explore the applicability of these language models for abstractive summarization. To that end, we fine-tune the GPT-2 architecture by casting the summarization task as a language modeling training objective; and we use the RoBERTa counterpart to warm-start the encoder and decoder of sequence-to-sequence models, which can be subsequently fine-tuned employing regular training procedures for sequence transduction tasks.

The trained models deliver competitive results, yielding higher ROUGE scores than the MarIA GPT-2 generative model in a zero-shot setting in all the experiments conducted. We believe this work provides the NLP community with a framework that could be extended to other mono-lingual language models, all with orders of magnitude less computational complexity with respect to the pre-training of encoder-decoder models.

---

</br>

## Performance

| Model                           | ROUGE-1   | ROUGE-2   | ROUGE-L   |
|---------------------------------|-----------|-----------|-----------|
| MultiSumm (Cao et al., 2020)    | 31.18     | 12.24     | 26.22     |
| NASes (Ahuir et al., 2021)      | 30.60     | 10.75     | 22.29     |
| mT5 (Hasan et al., 2021)        | **30.93** | **12.14** | **23.76** |
| MarIA GPT-2 BASE                | 21.02     | 4.37      | 17.26     |
| MarIA GPT-2 LARGE               | 22.68     | 5.39      | 18.63     |
| MarIA RoBERTa2RoBERTa BASE      | 21.24     | 4.74      | 16.62     |
| MarIA RoBERTa2RoBERTa LARGE     | 20.89     | 4.81      | 16.56     |
| Distilled mT5 (Fernández, 2022) | **28.66** | **8.80**  | **23.11** |
| MarIA GPT-2 BASE  zero-shot     | 19.35     | 3.63      | 16.26     |
| MarIA GPT-2 BASE                | 25.32     | 6.90      | 21.39     |
| MarIA GPT-2 LARGE               | 28.17     | 8.79      | 23.00     |
| MarIA RoBERTa2RoBERTa BASE      | 25.11     | 7.07      | 19.80     |
| MarIA RoBERTa2RoBERTa LARGE     | 23.67     | 6.49      | 18.91     | 


</br>

## Run the ES-Sum web application locally

### BASH
```
$ export FLASK_APP=app
$ flask run
```

### CMD
> set FLASK_APP=app
> flask run

### Powershell
```
> $env:FLASK_APP = "app"
> flask run
```
</br>

## Run the code

### Fine-tune the GPT-2 architecture
```
usage: gpt2_summarizer_train.py [-h] [--root_dir ROOT_DIR] [--model {base,large}] --batch_size BATCH_SIZE --num_train_epochs NUM_TRAIN_EPOCHS
                                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--max_grad_norm MAX_GRAD_NORM] [--lr LR]
                                [--n_gpu N_GPU] [--num_workers NUM_WORKERS] [--device {cuda,cpu}] [--do_eval] -o OUTPUT_DIR [--seed SEED]
                                --test_data_dir TEST_DATA_DIR [--max_length MAX_LENGTH] [--temperature TEMPERATURE] [--top_k TOP_K]
                                [--top_p TOP_P]

optional arguments:
  -h, --help            show this help message and exit
  --root_dir ROOT_DIR   Parent directory containing at least the training and validation datasets to fine tune the model. The data should be      
                        formatted in such way that it can be processed by a `GPT2SumDataset` object. Refer to the `prepare_data.py` script for    
                        further information
  --model {base,large}  Type of BSC GPT2 architecture
  --batch_size BATCH_SIZE
                        Training batch size
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Number of training epochs
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Accumulated gradients run integer K small batches of size N before doing a backward pass.
  --max_grad_norm MAX_GRAD_NORM
                        Max norm of the gradients
  --lr LR               Initial learning rate
  --n_gpu N_GPU         Number of GPUs available
  --num_workers NUM_WORKERS
                        Number of workers (CPUs) available
  --device {cuda,cpu}   torch.device object representing the device on which a torch.Tensor is or will be allocated.
  --do_eval             Assess performance on test set (located as a subdirectory of `root_dir` and named after "test") once the model has been   
                        trained.
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to save the trained model and the evaluation results
  --seed SEED           Initialization state of a pseudo-random number generator to grant reproducibility of the experiments
  --test_data_dir TEST_DATA_DIR
                        Parent directory containing the test dataset.
  --max_length MAX_LENGTH
                        Max summary length
  --temperature TEMPERATURE
                        Introduce randomness of the predictions by scaling the model logits before applying softmax
  --top_k TOP_K         Keep only top k tokens with highest probability (top-k filtering)
  --top_p TOP_P         Keep the top tokens with cumulative probability >= top_p (nucleus filtering)
```

### Evaluate the GPT-2 fine-tuned model 
```
usage: gpt2_summarizer_inference.py [-h] --train_data_dir TRAIN_DATA_DIR --test_data_dir TEST_DATA_DIR [--model {base,large}] --batch_size
                                    BATCH_SIZE --num_train_epochs NUM_TRAIN_EPOCHS [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]    
                                    [--max_length MAX_LENGTH] [--num_workers NUM_WORKERS] [--temperature TEMPERATURE] [--top_k TOP_K]
                                    [--top_p TOP_P] [--device DEVICE] -o OUTPUT_DIR

optional arguments:
  -h, --help            show this help message and exit
  --train_data_dir TRAIN_DATA_DIR
                        Parent directory containing the training dataset on which the model has been trained.
  --test_data_dir TEST_DATA_DIR
                        Parent directory containing the test dataset.
  --model {base,large}  Type of BSC GPT2 architecture
  --batch_size BATCH_SIZE
                        batch_size
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Number of training epochs
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Accumulated gradients run integer K small batches of size N before doing a backward pass.
  --max_length MAX_LENGTH
                        Max summary length
  --num_workers NUM_WORKERS
                        Number of workers (CPUs) available
  --temperature TEMPERATURE
                        Introduce randomness of the predictions by scaling the model logits before applying softmax
  --top_k TOP_K         Keep only top k tokens with highest probability (top-k filtering)
  --top_p TOP_P         Keep the top tokens with cumulative probability >= top_p (nucleus filtering)
  --device DEVICE       torch.device object
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        path to save the trained model and evaluation results
```

### Fine-tune the RoBERTa architecture

```
usage: roberta_encdec_train.py [-h] [--data_dir DATA_DIR] [--model {base,large}] [--tie_weights] --batch_size BATCH_SIZE --num_train_epochs
                               NUM_TRAIN_EPOCHS [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                               [--summary_min_length SUMMARY_MIN_LENGTH] [--summary_max_length SUMMARY_MAX_LENGTH] [--lr LR] -mo MODEL_DIR -o     
                               OUTPUT_DIR [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Parent directory containing at least the training and validation datasets to fine tune the model. The data should be      
                        formatted in such way that it can be processed by a `GPT2SumDataset` object. Refer to the `prepare_data.py` script for    
                        further information
  --model {base,large}  Type of BSC RoBERTa architecture
  --tie_weights         Tie encoder decoder weights
  --batch_size BATCH_SIZE
                        Training batch size
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Number of training epochs
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Accumulated gradients run integer K small batches of size N before doing a backward pass.
  --summary_min_length SUMMARY_MIN_LENGTH
                        Minimum length of the decoder output.
  --summary_max_length SUMMARY_MAX_LENGTH
                        Maximum length of the decoder output.
  --lr LR               Initial learning rate
  -mo MODEL_DIR, --model_dir MODEL_DIR
                        Directory to save the trained model (and intermediate checkpoints)
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Directory to save the trained model and the evaluation results
  --seed SEED           Initialization state of a pseudo-random number generator to grant reproducibility of the experiments
```

### Evaluate the RoBERTa2RoBERTa fine-tuned model 
```
usage: roberta_encdec_inference.py [-h] [--train_data_dir TRAIN_DATA_DIR] [--test_data_dir TEST_DATA_DIR] [--model {base,large}] [--tie_weights]
                                   --batch_size BATCH_SIZE --num_train_epochs NUM_TRAIN_EPOCHS
                                   [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--summary_min_length SUMMARY_MIN_LENGTH]        
                                   [--summary_max_length SUMMARY_MAX_LENGTH] [--train_summary_min_length TRAIN_SUMMARY_MIN_LENGTH]
                                   [--train_summary_max_length TRAIN_SUMMARY_MAX_LENGTH] [--temperature TEMPERATURE] [--top_k TOP_K]
                                   [--top_p TOP_P] [--num_beams NUM_BEAMS] -mo MODEL_DIR [--checkpoint_at_step CHECKPOINT_AT_STEP] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --train_data_dir TRAIN_DATA_DIR
                        Parent directory containing the training dataset on which the model has been trained.
  --test_data_dir TEST_DATA_DIR
                        Parent directory containing the test dataset.
  --model {base,large}  Type of BSC RoBERTa architecture
  --tie_weights         Tie encoder decoder weights
  --batch_size BATCH_SIZE
                        Training batch size
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Number of training epochs
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Accumulated gradients run integer K small batches of size N before doing a backward pass.
  --summary_min_length SUMMARY_MIN_LENGTH
                        Minimum length of the decoder output.
  --summary_max_length SUMMARY_MAX_LENGTH
                        Maximum length of the decoder output.
  --train_summary_min_length TRAIN_SUMMARY_MIN_LENGTH
                        Minimum length of the decoder output.
  --train_summary_max_length TRAIN_SUMMARY_MAX_LENGTH
                        Maximum length of the decoder output.
  --temperature TEMPERATURE
                        Introduce randomness of the predictions by scaling the model logits before applying softmax
  --top_k TOP_K         Keep only top k tokens with highest probability (top-k filtering)
  --top_p TOP_P         Keep the top tokens with cumulative probability >= top_p (nucleus filtering)
  --num_beams NUM_BEAMS
                        Number of beams in Beam search
  -mo MODEL_DIR, --model_dir MODEL_DIR
                        Directory to save the trained model (and intermediate checkpoints)
  --checkpoint_at_step CHECKPOINT_AT_STEP
                        Load a checkpoit at a specific training step
  --seed SEED           Initialization state of a pseudo-random number generator to grant reproducibility of the experiments
```

</br>

## Code dependences
```
torch>=1.10.2
tqdm==4.62.3
transformers==4.17.0
numpy>=1.19.5
tensorboard==2.8.0
pandas>=1.1.5
flask
bootstrap-flask
flask-debug
spacy
rouge
```

---

