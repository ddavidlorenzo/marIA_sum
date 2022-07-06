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

## Run the code

### Fine-tune the GPT-2 architecture

### Fine-tune the RoBERTa architecture
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

