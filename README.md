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
