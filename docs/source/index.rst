.. Implementation of a framework to fine-tune GPT/GPT2 based models on abstractive summarization. documentation master file, created by
   sphinx-quickstart on Wed Jun  9 10:43:12 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
=============
Abstractive summarization has experienced a surge of interest thanks to recent advancements on Transformer-based encoder-decoder models, with standout proposals like PEGASUS, that incorporates explicit pre-training objectives tailored for such a task, exhibiting unprecedented level of performance on natural language generation tasks. However, the humongous amount of data required and the massive computational cost attached to the pre-training of these architectures imposes a substantial burden on their availability in languages other than English, with the very exception of their multi-lingual homologous.

The recent large Spanish language models from the MarIA project, based on the RoBERTa and GPT-2 architectures, have shown promising results, pushing the state-of-the-art on multiple natural language understanding tasks. However, encoder- and decoder-only systems pose as an architecturally suboptimal approach to resolve sequence-to-sequence tasks. In this work, we explore the applicability of these language models for abstractive summarization.

In this page, we present the documentation of a fully-documented API-like tool to fine-tune GPT-like architectures on abstractive summarization.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   eda
   code
   

   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
