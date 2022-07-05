from functools import wraps
from pathlib import Path
from typing import Callable, Union
from transformers import AutoTokenizer
import torch
from dataset import GPT2SumDataset
from utils import makedir, prepare_input_summarizer
import torch.nn.functional as F
from tqdm import trange

class GPT2Summarizer:
    """This class serves as a common interface for GPT-2 based fine-tuned architectures
    for abstractive summarization, both at training and inference time. 
    
    Notes
    ------
    This class is not meant to be instantiated - use instead the specialized subclasses defined for
    training and inference accordingly.
    """

    # Min valid value for the `num_workers` property
    _MIN_NUM_WORKERS = 1
    # Min valid batch size
    _MIN_BATCH_SIZE = 1
    # Min valid number of epochs
    _MIN_NUM_TRAIN_EPOCHS = 1
    # Min valid number of gradient accumulation steps
    _MIN_GRADIENT_ACCUMULATION_STEPS = 1
    # Per article/summary separator token id
    _sep_token_id = None
    # Pad special token id
    _pad_token_id = None


    def __init__(self,
                 checkpoint_name:str,
                 batch_size:int,
                 num_train_epochs:int,
                 gradient_accumulation_steps:int,
                 num_workers:int,
                 device:torch.device,
                 output_dir:Union[str, Path]
                ) -> None:
        """Constructor of a `GPT2Summarizer` instance. This method is solely meant to be
        invoked by the subclasses implementing this interface. 

        Parameters
        ----------
        checkpoint_name : str
            Model id of a pretrained HuggingFace Transformer hosted inside a model repo on 
            huggingface.co 
        batch_size : int
            Training batch size
        num_train_epochs : int
            Number of training epochs
        gradient_accumulation_steps : int
            Number of gradient accumulation steps
        num_workers : int
            Number of workers available
        device : torch.device
            torch.device object representing the device on which a torch.Tensor is or 
            will be allocated.
        output_dir : Union[str, Path]
            Output directory whereby outputs generated at training are stored.
        """
        # Set output directory
        self.output_dir = output_dir 
       
        # Setup tokenizer from checkpoint       
        self.tokenizer = checkpoint_name
        # Separator token id       
        self._sep_token_id = self.tokenizer.sep_token_id
        # Padding token id
        self._pad_token_id = self.tokenizer.pad_token_id

        # Setup allocation device for torch tensors
        self._device = device
        # Set number of workers
        self.num_workers = num_workers
        # Set batch size
        self.batch_size = batch_size
        # Set number of training epochs
        self.num_train_epochs = num_train_epochs
        # Set number of gradient accumulation steps
        self.gradient_accumulation_steps = gradient_accumulation_steps


    @property
    def batch_size(self) -> int:
        """Training batch size, i.e., number of samples processed in parallel by
        the model. 

        Returns
        -------
        int
            Training batch size
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size:int) -> None:
        """Setter method for the training batch size.

        Parameters
        ----------
        batch_size : int
            Size of the training batch size

        Raises
        ------
        ValueError
            Raised for illegal values of batch size, i.e., those smaller than 
            `_MIN_BATCH_SIZE`
        """
        if batch_size < self._MIN_BATCH_SIZE:
            raise ValueError(f'Batch size must be at least {self._MIN_BATCH_SIZE}')
        self._batch_size=batch_size

    @property
    def gradient_accumulation_steps(self) -> int:
        """Number of K mini-batches of size `batch_size` to run before performing a backward
        pass.

        Returns
        -------
        int
            Number of gradient accumulation steps
        """
        return self._gradient_accumulation_steps

    @gradient_accumulation_steps.setter
    def gradient_accumulation_steps(self, gradient_accumulation_steps:int) -> None:
        """Setter method for the number of gradient accumulation steps.

        Parameters
        ----------
        gradient_accumulation_steps : int
            Number of gradient accumulation steps

        Raises
        ------
        ValueError
            Raised for illegal values of `gradient_accumulation_steps`, i.e., those smaller than 
            `_MIN_GRADIENT_ACCUMULATION_STEPS`.
        """
        if gradient_accumulation_steps < self._MIN_GRADIENT_ACCUMULATION_STEPS:
            raise ValueError(f'The number of gradient accumulation steps must be at least {self._MIN_GRADIENT_ACCUMULATION_STEPS}')
        self._gradient_accumulation_steps = gradient_accumulation_steps
   
    @property
    def num_train_epochs(self) -> int:
        """Number of training epochs, i.e., number of complete passes through the entire
         training dataset. 

        Returns
        -------
        int
            Number of training epochs
        """
        return self._num_train_epochs

    @num_train_epochs.setter
    def num_train_epochs(self, num_train_epochs:int) -> None:
        """Setter method for the number of training epochs.

        Parameters
        ----------
        num_train_epochs : int
            Number of training epochs

        Raises
        ------
        ValueError
            Raised for illegal values of `num_train_epochs`, i.e., those smaller than 
            `_MIN_NUM_TRAIN_EPOCHS`.
        """
        if num_train_epochs < self._MIN_NUM_TRAIN_EPOCHS:
            raise ValueError(f'The number of epochs for training must be at least {self._MIN_NUM_TRAIN_EPOCHS}')
        self._num_train_epochs = num_train_epochs

    @property
    def num_workers(self) -> int:
        """ Number of processing elements (typically in terms of number of CPU cores 
        available) at your disposal to process data loading in parallel. In practice,
        `num_workers` equals the number of samples that can be loaded in parallel. 

        Returns
        -------
        int
            Number of workers
        
        Notes
        --------
        Multi-process data loading (pytorch): https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        """
        return self._num_workers

    @num_workers.setter
    def num_workers(self, num_workers:int) -> None:
        """Setter method for the number of workers available.

        Parameters
        ----------
        num_workers : int
            Number of workers available

        Raises
        ------
        ValueError
            Raised for illegal values of `num_workers`, i.e., those smaller than 
            `_MIN_NUM_WORKERS`.
        """
        if num_workers < self._MIN_NUM_WORKERS:
            raise ValueError(f'The number of workers must be at least {self._MIN_NUM_WORKERS}')
        self._num_workers=num_workers

    @property
    def output_dir(self) -> Path:
        """Path of the directory of the generated model and training statistics. 

        Returns
        -------
        Path
            Directory whereby the trained models, along with their configuration and
            other statistics are allocated
        """
        return self._output_dir

    @output_dir.setter
    def output_dir(self, path:Union[str, Path]) -> None:
        """Setter method for the output directory of the class object.
         * At *training* time, the trained models (`bin` format), along with their 
         configuration (in `json` format) and some statistics in the training process 
         will be allocated in directory `path`, if it is a valid directory.
         * At *inference* time, trained models and their configuration will be loaded
         from this directory. 

        Parameters
        ----------
        path : Union[str, Path]
            Path to the desired directory to store/load trained models, config files 
            and/or training status.
        """
        self._output_dir = Path(path)
        makedir(self._output_dir)

    @property
    def tokenizer(self):
        """HuggingFace Tokenizer object, which is targeted at preparing the inputs for a model.
        By default, the tokenizer to use is that of the checkpoint (pre-trained model)

        Returns
        -------
        Any
            Tokenizer for the model

        Notes
        --------
        Documentation of HuggingFace Tokenizer class: https://huggingface.co/docs/transformers/main_classes/tokenizer
        """
        try: 
            return self._tokenizer
        except AttributeError:
            print("Setting tokenizer")
            # Ignore padding token for loss calculation
            self.tokenizer = self.model.name_or_path
            return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, name_or_path:Union[str, Path]) -> None:
        """ Setter method for the model tokenizer after adding the separator and 
        padding special tokens to fine tune the model for abstractive summarization.

        Parameters
        ----------
        name_or_path : Union[str, Path]
            Model identifier of a predefined tokenizer hosted inside a model repo
             on huggingface.co
        """
        self._tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        special_tokens = dict(
            pad_token = '<|pad|>',
            sep_token = '<|sep|>'
        ) 
        _ = self._tokenizer.add_special_tokens(special_tokens)

    @property
    def _config_file(self):
        """Filepath whereby the configuration file of a fine-tuned model is to be loaded/stored
        from.

        Raises
        ------
        NotImplementedError
            To be implemented by all subclasses.
        """
        raise NotImplementedError("This method must be implemented")

    @property
    def _model_file(self):
        """Filepath whereby a model is to be loaded/store from.

        Raises
        ------
        NotImplementedError
            To be implemented by all subclasses.
        """
        raise NotImplementedError("This method must be implemented")

    def attrs_to_str(self):
        """Meaningful string-like representation of the training attributes to fine-tune the model,
        serving as a straightforward and effective fashion to uniquely identify the model. 

        Raises
        ------
        NotImplementedError
            To be implemented by all subclasses.
        """
        raise NotImplementedError("This method must be implemented")

    def sample_sequence(self, context, length:int, temperature=1, top_k=0, top_p=0.0) -> torch.Tensor:
        """Generate `length` new tokens based on a context (`context`).

        Parameters
        ----------
        context : array-like
            Context tokenized text
        length : int
            Number of tokens to generate
        temperature : int, optional
            Introduce randomness of the predictions by scaling the model logits before
            applying softmax, by default 1. Values for temperature range in (0,1], where
            values closer to 1 indicate less randomness
        top_k : int, optional
            Perform top-k filtering (only if `top_k` > 0), by default 0
        top_p : float, optional
            Perform nucleus filtering (only if `top_p` > 0), by default 0.0

        Returns
        -------
        torch.Tensor
            Tensor containing the tokenized text of the context and the generated tokens

        Notes
        ------
        Original code by Thomas Wolf:

        (1) https://github.com/huggingface/transformers/blob/5c3b32d44d0164aaa9b91405f48e53cf53a82b35/examples/run_generation.py
        """
        # Create tensor from context object
        context = torch.tensor(context, dtype=torch.long, device=self._device)
        # Flatten tensor
        context = context.unsqueeze(0)
        generated = context
        # Disable gradient computation
        with torch.no_grad():
            # Generate `length` tokens.
            for _ in trange(length):
                inputs = dict(input_ids = generated)
                # Yield logits from the inputs to the model
                outputs = self.model(**inputs)
                # Scale the logits according to `temperature` to control the randomness of predictions
                next_token_logits = outputs[0][0, -1, :] / temperature
                # Filter logits according to the prob of the k-th token (top_k) and/or according to 
                # the cumulative probability of the tokens in the logits (top_p)
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample one token from the normalized filtered logits.
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                # Concatenate the token to current "context".
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        return generated

    def top_k_top_p_filtering(self, logits:torch.Tensor, top_k=0, top_p=0.0, filter_value=-float('Inf')) -> torch.Tensor:
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        
        Parameters
        ----------
        logits : torch.Tensor
            Logits distribution shape (vocabulary size)
        top_k : int, optional
            Keep only top k tokens with highest probability (top-k filtering), by default 0. 
            Top-k filtering is performed for `top_k` > 0
        top_p : float, optional
            Keep the top tokens with cumulative probability >= top_p (nucleus filtering), by default 0.0.
            Nucleus filtering is performed for `top_p` > 0
        filter_value : float, optional
            Logits filter value, by default -float('Inf')

        Returns
        -------
        torch.Tensor
            Filtered distribution of logits

        Notes
        ------
        Original code by Thomas Wolf:

        (1) https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

        (2) https://github.com/huggingface/transformers/blob/5c3b32d44d0164aaa9b91405f48e53cf53a82b35/examples/run_generation.py

        """
        # Filtering is done in a one sample at a time basis (batch size at generation time is 1).
        assert logits.dim() == 1
        # The number of k tokens to keep is the minimum between `top_k` and the total number of tokens
        # available.
        top_k = min(top_k, logits.size(-1))
        
        # Top-k filtering is performed if `top_k` > 0
        if top_k > 0:
            # Tokens with a probability less that that of the k-th token are removed.
            to_del_ids = logits < torch.topk(logits, top_k)[0][..., -1, None]
            # To that end, their logits are set to an arbitrarily low value, e.g., -Inf
            logits[to_del_ids] = filter_value

        # Nucleus filtering is performed if `top_p` > 0
        if top_p > 0.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # And compute the cumulative probability of the normalized logits
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_to_del_ids = cumulative_probs > top_p
            
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_to_del_ids[..., 1:] = sorted_to_del_ids[..., :-1].clone()
            sorted_to_del_ids[..., 0] = 0

            # Keep tokens with a cumulative prob below the threshold
            to_del_ids = sorted_indices[sorted_to_del_ids]
            # To that end, set those that have a greater probability to an arbitrarily low value, 
            # e.g., -Inf
            logits[to_del_ids] = filter_value
        # Return the filtered logits
        return logits

    def tokenize_input(func:Callable) -> Callable:
        """Ensure text is tokenized before feeding it into the model for summary generation.

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
        def inner(self, context, max_length, **kwargs):
            # If context is not tokenized or it does not fit into the model taking into account
            # the desired length for the summary...
            if type(context) == str or len(context) + max_length >= self.model.config.n_positions:
                context = prepare_input_summarizer(context, 
                                                   self.tokenizer, 
                                                   max_input=self.model.config.n_positions,
                                                   gpt2_summary_length=max_length,
                                                   as_tokens=True)[0]
                # Append separator token
                context.append(self._sep_token_id)
            return func(self, context, max_length, **kwargs)
        return inner

    def beam_search(self, context, max_length=60, beam_size=4, temperature=1):
        """Performs beam search from `context`, generating up to `max_length` tokens and
        keeping `beam_size` hypotheses at each generation step.

        Parameters
        ----------
        context : array-like
            Context tokenized text
        max_length : int, optional
            Maximum length, in terms of tokens, of the generated summary, by default 60
        beam_size : int, optional
            Keep the most likely `beam_size` of hypotheses at each generation step to 
            eventually choose the hypothesis that has the overall highest probability,
            by default 4
        temperature : int, optional
            Introduce randomness of the predictions by scaling the model logits before
            applying softmax, by default 1. Values for temperature range in (0,1], where
            values closer to 1 indicate less randomness

        Returns
        -------
            `beam_size` generated sequences along with their respective scores

        Notes
        ------
        Original code by Rohit Kumar Singh:

        (1) https://github.com/SKRohit/Generating_Text_Summary_With_GPT2/blob/master/utils.py
        """
        context = torch.tensor(context, dtype=torch.long, device=self._device)
        context = context.unsqueeze(0)
        with torch.no_grad():  
            outputs = self.model(input_ids=context) 
            next_token_logits = outputs[0][0, -1, :] / temperature
            next_token_probs = F.softmax(next_token_logits)
            scores, indices = torch.topk(next_token_probs, beam_size)
            indices = indices.tolist()
            sequences = [[c] for c in indices]
            for _ in trange(max_length-1):
                logits = torch.zeros(beam_size*len(next_token_logits))
                for j in range(len(sequences)):
                    new_generated = torch.cat((context,torch.tensor([sequences[j]], dtype=torch.long, device=self._device)),dim=1)
                    outputs = self.model(input_ids=new_generated) 
                    next_token_logits = outputs[0][0, -1, :] / temperature
                    next_token_probs = F.softmax(next_token_logits)
                    start, stop = j*len(next_token_logits), (j+1)*len(next_token_logits)
                    logits[start:stop] = scores[j]*next_token_probs
                scores, new_logits_indices = torch.topk(logits,beam_size)
                logits = (new_logits_indices%len(self.tokenizer)).tolist()
                for j in range(len(sequences)):
                    sequences[j] = sequences[j]+[logits[j]]
        return scores, sequences

    @tokenize_input
    def generate_sample(self, context, max_length=60, temperature=1, top_k=10, top_p=0.5) -> str:
        """Generate summary from `context` with a maximum length of `max_length` tokens

        Parameters
        ----------
        context : array-like
            Context tokenized text
        max_length : int, optional
            Maximum length, in terms of tokens, of the generated summary, by default 60
        temperature : int, optional
            Introduce randomness of the predictions by scaling the model logits before
            applying softmax, by default 1. Values for temperature range in (0,1], where
            values closer to 1 indicate less randomness
        top_k : int, optional
            Perform top-k filtering (only if `top_k` > 0), by default 10
        top_p : float, optional
            Perform nucleus filtering (only if `top_p` > 0), by default 0.5

        Returns
        -------
        str
            Generated summary in plain text
        """
        # Generate new `max_length` tokens
        generated_text = self.sample_sequence(context, max_length, temperature, top_k, top_p)
        # Extract generated summary (omit the context tokens)
        generated_text = generated_text[0, len(context):].tolist()
        # Decode tokens, skipping special tokens such as the padding or separator token
        text = self.tokenizer.convert_ids_to_tokens(generated_text,skip_special_tokens=True)
        text = self.tokenizer.convert_tokens_to_string(text)
        # Remove prefix token
        return text.replace("</s>", "")

    @tokenize_input
    def generate_beam_sample(self, context, max_length=60, beam_size=4, temperature=1):
        """ Generate summary from `context` with a maximum length of `max_length` tokens using
        beam search.

        Parameters
        ----------
        context : array-like
            Context tokenized text
        max_length : int, optional
            Maximum length, in terms of tokens, of the generated summary, by default 60
        beam_size : int, optional
            Keep the most likely `beam_size` of hypotheses at each time step to eventually
            choose the hypothesis that has the overall highest probability, by default 4
        temperature : int, optional
            Introduce randomness of the predictions by scaling the model logits before
            applying softmax, by default 1. Values for temperature range in (0,1], where
            values closer to 1 indicate less randomness

        Returns
        -------
        _type_
            `beam_size` generated sequences sorted in decreasing order of score (larger 
            scores signify better hipothetical quality of summary)
        """
        scores, sequences = self.beam_search(context, max_length, beam_size=beam_size, temperature=temperature)
       
        # Extract generated summary (omit the context tokens)
        generated_texts = [self.tokenizer.decode(gen, skip_special_tokens=True).replace('</s>', '') for gen in sequences]
        
        generated_texts = sorted(list(zip(scores, generated_texts)), key=lambda x:x[0], reverse=True)

        return generated_texts

    @tokenize_input
    def generate_sample_huggingface(self, context, max_length=60, **gen_kwargs):
        """Generate summary from `context` with a maximum length of `max_length` tokens using
        HuggingFace's functions for text generation

        Parameters
        ----------
        context : array-like
            Context tokenized text
        max_length : int, optional
            Maximum length, in terms of tokens, of the generated summary, by default 60

        Returns
        -------
        str
            Generated summary in plain text
        """
        inputs = torch.tensor([context])
        print(inputs.shape)
        # inputs = self.tokenizer.encode(context, return_tensors='pt')

        outputs = self.model.generate(inputs, max_length=max_length, **gen_kwargs)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        

    def generate_summaries_from_dataset(self,
                                        test_data_dir:GPT2SumDataset,
                                        max_length=100,
                                        temperature=1,
                                        top_k=10,
                                        top_p=0.5,
                                        out_path:Union[str, Path]=None
                                        ) -> None:
        """ Generate summaries from test data and store them in disk.

        Parameters
        ----------
        test_data_dir : array-like
            Test data from which samples are withdrawn
        max_length : int, optional
            Maximum length, in terms of tokens, of the generated summary, by default 100
        temperature : int, optional
            Introduce randomness of the predictions by scaling the model logits before
            applying softmax, by default 1. Values for temperature range in (0,1], where
            values closer to 1 indicate less randomness
        top_k : int, optional
            Perform top-k filtering (only if `top_k` > 0), by default 0
        top_p : float, optional
            Perform nucleus filtering (only if `top_p` > 0), by default 0.0
        out_path : Union[str, Path], optional
            Custom filepath to store the generated summaries, by default None
        """
        # by default, the summaries generated are stored in a sibling directory to that of 
        # the data called "inference"
        if not out_path:
            test_data_name = Path(test_data_dir).name
            test_data_name = "all_" if "all" in test_data_name else test_data_name + "_"
            out_path = Path(Path(test_data_dir).parent, 'inference', f'generated_summaries_{self.attrs_to_str(add=f"temperature_{int(temperature)}_topk_{top_k}_topp_{top_p}_max_length_{max_length}", test_data_name=test_data_name)}.txt')

        print('Generated summaries will be stored at', out_path)

        dataset = GPT2SumDataset(test_data_dir, self.model.config.n_positions, self._sep_token_id, self._pad_token_id,  mode='test')
        summaries = []
        with open(out_path, 'w+') as f:
            for i in range(len(dataset)):
                # sample data sequentially
                sample = dataset[i]
                # article/summary separator index
                idx = sample['sum_idx']
                # get article from sample
                article = sample['article'][:idx].tolist()
                # Dump generated sumple to output file
                generated_sample = self.generate_sample(article, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p)
                summaries.append(generated_sample)
                f.write(f'{generated_sample}\n')
                print(i)
        return summaries

    def log_generated_summaries(self, data, num=1, eval_step=False, max_length=60, temperature=1, top_k=10, top_p=0.5) -> None:
        """Log `num` generated summaries from `data`.

        Parameters
        ----------
        data : array-like
            Dataset from which samples are withdrawn
        num : int, optional
            number of summaries to generate (and log), by default 1
        eval_step : bool, optional
            Whether to log the article and actual summary, by default False
        max_length : int, optional
            Maximum length, in terms of tokens, of the generated summary, by default 100
        temperature : int, optional
            Introduce randomness of the predictions by scaling the model logits before
            applying softmax, by default 1. Values for temperature range in (0,1], where
            values closer to 1 indicate less randomness
        top_k : int, optional
            Perform top-k filtering (only if `top_k` > 0), by default 10
        top_p : float, optional
            Perform nucleus filtering (only if `top_p` > 0), by default 0.5

        Notes
        ------
        Code adaptation by Rohit Kumar's work:

        (1) https://github.com/SKRohit/Generating_Text_Summary_With_GPT2/blob/master/utils.py
        """
        # Generate `num` new summaries
        for i in range(num):
            # sample data sequentially
            sample = data[i]
            # article/summary separator index
            idx = sample['sum_idx']
            # get article from sample
            article = sample['article'][:idx].tolist()
            # Generate sample
            generated_text = self.generate_sample(article, 
                                                  max_length=max_length,
                                                  temperature=temperature,
                                                  top_k=top_k,
                                                  top_p=top_p)
            # Log results
            if eval_step==False:
                print('Article', end='\n\n')
                print(self.tokenizer.decode(article), end='\n\n')
                print("Generated summary", end='\n\n')
                print(generated_text, end='\n\n')
                print('Actual summary', end='\n\n')
                summary = sample['article'][idx+1:][:max_length].tolist()
                print(self.tokenizer.decode(summary), end='\n\n')
            else:
                print("Generated summary", end='\n\n')
                print(generated_text, end='\n\n')
