import argparse
from pathlib import Path
import time
from typing import Callable, Tuple, Union
import pandas as pd

from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import trange, tqdm

from dataset import GPT2SumDataset
from gpt2_summarizer import GPT2Summarizer 
from utils import set_seed, format_time, compute_rouge_score


class TrainGPT2Summarizer(GPT2Summarizer):
    """This class provides a basic interface to train a GPT-2 based pre-trained model for
    abstractive summarization, whereby fine-tuning can be simply achieved by instantiating
    a class object and subsequently invoking the `train` function.
    """

    def __init__(self,
                 checkpoint_name:str,
                 data_dir:Union[str, Path],
                 batch_size:int,
                 num_train_epochs:int,
                 gradient_accumulation_steps:int,
                 max_grad_norm:float,
                 lr:float,
                 n_gpu:int,
                 num_workers:int,
                 device:torch.device,
                 output_dir:Union[str, Path],
                 seed:int
                ) -> None:
        """Constructor of a `TrainGPT2Summarizer` instance, which provides all necessary functionality
        to allow for the fine-grained tuning of a GPT2-like architecture for abstractive summarization.
        A prior step to train any model is to have data well formatted. To that end, please refer to 
        the `prepare_data.py` module and its documentation, should you require it.

        Parameters
        ----------
        checkpoint_name : str
            Model id of a pretrained HuggingFace Transformer hosted inside a model repo on 
            huggingface.co 
        data_dir : Union[str, Path]
            Parent directory containing at least the training and validation datasets to fine
            tune the model. The data should be formatted in such way that it can be processed
            by a `GPT2SumDataset` object. Refer to the `prepare_data.py` script for further 
            information.
        batch_size : int
            Training batch size
        num_train_epochs : int
            Number of training epochs
        gradient_accumulation_steps : int
            Number of gradient accumulation steps
        max_grad_norm : float
            Max norm of the gradients. This helps leveraging the exploding gradients problem, whereby
            large gradient vectors are rescaled so that their norm is at most `max_grad_norm`
        lr : float
            Initial learning rate
        n_gpu : int
            Number of GPUs available
        num_workers : int
            Number of workers available
        device : torch.device
            torch.device object representing the device on which a torch.Tensor is or 
            will be allocated.
        output_dir : Union[str, Path]
            Output directory whereby outputs generated at training are stored.
        seed : int
            Initialization state of a pseudo-random number generator to grant reproducibility of
            the experiments
        """
        # Setup some of the parameters using the constructor method of the base class.
        super().__init__(
            checkpoint_name=checkpoint_name,
            batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_workers=num_workers,
            device=device,
            output_dir=output_dir    
        )
        # in practice, the number of GPUs available is utilized to control
        # randomness in a setting subject to parallelized/distributed execution
        self._n_gpu = n_gpu
        self._seed = seed
        # Set the max norm of the gradients
        self.max_grad_norm = max_grad_norm
        # Initial learning rate
        self.lr = lr

        # Load a GPT2 Model transformer with a language modeling head on top (linear
        # layer with weights tied to the input embeddings).
        self.model = GPT2LMHeadModel.from_pretrained(checkpoint_name)            
        # Maximum input size for fine-tuning is that of the model
        max_input = self.model.config.n_positions
        
        # Setup training dataset
        self.train_dataset = GPT2SumDataset(data_dir, max_input, self._sep_token_id, self._pad_token_id, mode='train')
        
        # _train_data_name is used to generate output filenames automatically
        dir_name = Path(data_dir).name
        self._train_data_name = "all"  if "all" in dir_name else dir_name

        # Setup validation dataset
        self.val_dataset = GPT2SumDataset(data_dir, max_input, self._sep_token_id, self._pad_token_id, mode='val')

        # Resize input token embeddings matrix of the model in order to account for
        # the new tokens included: the separator and the padding token.
        self.model.resize_token_embeddings(len(self.tokenizer))
        # Send model parameters to the appropriate device (e.g., CPU for "cpu" or
        # GPU for "cuda").
        self.model.to(self._device)


    @property
    def loss_func(self) -> Callable:
        """Function that computes the cross entropy loss between input and target,
        ignoring the padding token.

        Returns
        -------
        Callable
            Cross entropy loss function
        """
        try: 
            return self._loss_func
        except AttributeError:
            print(f"Setting CrossEntropyLoss loss function with ignore_index={self._pad_token_id}")
            # Ignore padding token for loss calculation
            self._loss_func = CrossEntropyLoss(ignore_index=self._pad_token_id)
            return self._loss_func

    @property
    def max_grad_norm(self) -> float:
        """Max norm of the gradients. This helps leveraging the exploding gradients 
        problem, whereby large gradient vectors are rescaled so that their norm is at
        most `max_grad_norm`.

        Returns
        -------
        float
            Max norm of the gradients
        """
        return self._max_grad_norm

    @max_grad_norm.setter
    def max_grad_norm(self, max_grad_norm:float) -> None:
        """Setter method for the max norm of the gradients.

        Parameters
        ----------
        max_grad_norm : float
            Max norm of the gradients
        """
        self._max_grad_norm = max_grad_norm

    @property
    def _config_file(self) -> Path:
        """Output filepath of the configuration file (in `json` format) of the trained model. This file
        is dumped to the output directory (refer to `self.output_dir` prop.), and named after the "named"
        parameters utilized at training (refer to `self.attrs_to_str()` module). 

        Returns
        -------
        Path
            Output filepath for the configuration file of the fine-tuned model
        """

        return Path(self.output_dir, f'config_{self.attrs_to_str()}.json')

    @property
    def _model_file(self) -> Path:
        """Output filepath for the trained model binary (in `bin` format). This file is dumped to the 
        output directory (refer to `self.output_dir` prop.), and named after the "named" parameters 
        utilized at training (refer to `self.attrs_to_str()` module). 

        Returns
        -------
        Path
            Output filepath for the trained model binary
        """
        return Path(self.output_dir, f'model_{self.attrs_to_str()}.bin')

    def train(self,num_warmup_steps=200, num_training_steps=80000) -> pd.DataFrame:
        """Train a GPT-like architecture to generate abstractive summaries utilizing the AdamW
        (Decoupled Weight Decay Regularization) optimizer, gradient clipping, cross-entropy loss
        and a linear scheduler, with a learning rate that decreases linearly from the initial lr 
        set in the optimizer to 0, after a warmup period (`num_warmup_steps`) and during `num_training_steps`
        training steps, where the learning rate increases from 0 to the initial lr set in the 
        optimizer.

        Parameters
        ----------
        num_warmup_steps : int, optional
            Number of steps for the warmup phase of the linear scheduler, by default 100
        num_training_steps : int, optional
            Total number of training steps for the linear scheduler, by default 80000

        Returns
        -------
        pd.DataFrame
            Statistics generated throughout the training process arranged in a DataFrame

        Notes
        ------
        The code for this method is largely based on the following work:
        (1) https://mccormickml.com/2019/07/22/BERT-fine-tuning/#43-training-loop
        (2) https://colab.research.google.com/github/kozodoi/website/blob/master/_notebooks/2021-02-19-gradient-accumulation.ipynb#scrollTo=ISFvH2p8dqYQ
        (3) https://github.com/SKRohit/Generating_Text_Summary_With_GPT2/blob/master/train_gpt2_summarizer.py
        """
        # Sample elements from the training dataset without replacement
        train_sampler = RandomSampler(self.train_dataset)
        # Provides an iterable over the training dataset according to a sampling criteria 
        # (e.g., random sampling)
        train_dl = DataLoader(self.train_dataset,sampler=train_sampler,batch_size=self.batch_size,num_workers=self.num_workers)
        # Instantiate AdamW optimizer with model parameters and the original learning rate
        optimizer = AdamW(self.model.parameters(),lr=self.lr)
        # Instantiate linear scheduler with `num_warmup_steps` warmup steps and `num_training_steps`
        # training steps
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=num_warmup_steps,num_training_steps=num_training_steps)

        # Refers to the number of times the weights have been updated. If gradient accumulation
        # steps is 1, then global_step == number of batches processed by the model.
        global_step = 0
        # Training loss
        tr_loss = 0.0
        # Initially, set all gradients of the model to zero (clear model params)
        self.model.zero_grad()
        # Training iterator
        train_iterator = trange(int(self.num_train_epochs), desc="Epoch")
        # Set seed for either a sequential (e.g., "CPU" or "#GPU == 1") setting or in a
        # setting subject to parallelism ("#GPU > 1")
        set_seed(self._seed, self._n_gpu > 0)
        # List containing some useful per-epoch statistics
        history = []

        for epoch, _ in enumerate(train_iterator):
            epoch_iterator = tqdm(train_dl, desc="Training")
            # ith epoch training loss
            tr_epoch_loss = 0
            # ith epoch validation loss
            val_epoch_loss = 0
            # ith epoch model perplexity
            model_epoch_perplexity = 0
            # Number of times the model has been evaluated during the ith epoch
            n_evals = 0
            # Measure how long the training epoch takes.
            t0 = time.time()
            t0_val = 0
            
            for step, batch in enumerate(epoch_iterator):
                inputs, labels = batch['article'].detach(), batch['article'].detach()
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)

                # Sets model in training mode:
                # - normalization layers use per-batch statistics
                # - activates Dropout layers (if any)
                self.model.train()

                # Perform a forward pass (evaluate the model on this training batch).
                # It returns different numbers of parameters depending on what arguments
                # args given and what flags are set. For our usage here, it returns
                # logits, i.e., the model raw outputs prior to activation, but NOT the
                # loss - which is to be computed next.
                logits = self.model(inputs)[0]
                
                # Compute loss
                loss = self.compute_loss(logits, labels, batch['sum_idx'])

                # Calculate the average loss over all of the batches.
                loss = loss/self.gradient_accumulation_steps

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to "max_grad_norm", by default, 1.0.
                # This helps preventing the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                tr_loss += loss.item()
                # Accumulate the training loss for this epoch
                tr_epoch_loss += loss.item()

                # Update the weights of the model. This is done every 
                # `self.gradient_accumulation_steps` batches.
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Update parameters and take a step using the computed gradient.
                    # The optimizer dictates the "update rule"--how the parameters are
                    # modified based on their gradients, the learning rate, etc.
                    optimizer.step()
                    # Update learning rate schedule
                    scheduler.step()
                    
                    # Clear previously calculated gradients before performing a
                    # backward pass. PyTorch doesn't do this automatically because 
                    # accumulating the gradients is convenient while training deep
                    # recurrent networks. 
                    # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                    self.model.zero_grad()
                    # Update the counter for the number of weight updates.
                    global_step += 1
                    # Log loss value
                    print("loss:", loss.item())
                    print("lr:", scheduler.get_last_lr()[0], end="\n\n")

                    if (step + 1)/self.gradient_accumulation_steps == 1.0:
                        print('After 1st update: ', end='\n\n')
                        self.log_generated_summaries(self.val_dataset, eval_step=False)
                
                # Run evaluation on validation dataset every 10 weights updates
                # Adapted from:
                # https://colab.research.google.com/github/kozodoi/website/blob/master/_notebooks/2021-02-19-gradient-accumulation.ipynb#scrollTo=ISFvH2p8dqYQ
                if (step + 1) % (10*self.gradient_accumulation_steps) == 0:
                    # Update the number of evaluations run
                    n_evals += 1
                    # Run evaluation and yield the perplexity and loss on validation
                    # set, along with the time elapsed to run the evaluation.
                    eval_perplexity, eval_avg_loss, eval_time = self.eval()

                    # Accumulate the per-epoch model perplexity, validation loss
                    # and validation time.
                    model_epoch_perplexity += eval_perplexity
                    val_epoch_loss += eval_avg_loss
                    t0_val += eval_time
                    # Generate a sample from the validation dataset (just for logging
                    # purposes)
                    print('After', global_step+1,'updates: ', end='\n\n')
                    self.log_generated_summaries(self.val_dataset, eval_step=True)

            # Append statistics of the epoch, including the training loss,
            # the validation loss, the model perplexity and the training and
            # validation time.
            history.append(
                {
                    'epoch': epoch + 1,
                    'Training Loss': tr_epoch_loss/len(epoch_iterator),
                    'Valid. Loss': val_epoch_loss/n_evals,
                    'Perplexity': model_epoch_perplexity/n_evals,
                    'Training Time': format_time(time.time() - t0),
                    'Validation Time': format_time(t0_val)
                }
            )

        # Dump and return the generated training statistics
        return self._save_train_stats(history)

    def eval(self) -> Tuple[torch.Tensor, float, float]:
        """Evaluate performance of the model on the validation set

        Returns
        -------
        Tuple[torch.Tensor, float, float]
            Perplexity of the model, average loss and elapsed time
        """
        # At validation/inference time, the order of the samples is not relevant
        # because the parameters of the model remain the same. Thus we can simply
        # use a sequential sampler.
        eval_sampler = SequentialSampler(self.val_dataset)
        # Similarly to the training process, we use a data loader which automatically
        # manages a iterable over the validation samples.
        eval_dataloader = DataLoader(self.val_dataset, sampler=eval_sampler, batch_size=self.batch_size)
        # Validation loss
        eval_loss = 0.0
        # Number of evaluation steps
        nb_eval_steps = 0
        # Elapsed time
        eval_time = time.time()

        # Put the model in evaluation mode:
        # - deactivate normalization layers
        # - deactivate dropout layers
        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs, labels = batch['article'].detach().to(self._device), batch['article'].detach().to(self._device)
            
            # Turn off gradients computation
            with torch.no_grad():
                logits = self.model(inputs)[0]
                # Compute loss as for training samples
                lm_loss = self.compute_loss(logits, labels, batch['sum_idx'])
                # Update validation loss
                eval_loss += lm_loss.mean().item()
            # Update no. validation steps
            nb_eval_steps += 1

        # Compute avg validation loss
        eval_loss = eval_loss / nb_eval_steps
        # Compute model perplexity
        perplexity = torch.exp(torch.tensor(eval_loss))
        # Compute validation elapsed time
        eval_time = time.time() - eval_time
        
        # Log validation stats
        print(f"perplexity: {perplexity.item()}", end='\n')
        print(f"val. loss: {eval_loss}", end='\n')
        print(f"val. elapsed time: {eval_time}", end='\n')

        return perplexity, eval_loss, eval_time               

    def attrs_to_str(self, add:str=None, test_data_name="") -> str:
        """Yield a string-like representation of the training attributes to fine-tune the model,
        serving as a straightforward and effective fashion to uniquely identify the model. 
        
        Parameters
        ----------
        add : str, optional
            Substring to append at the end of the string model descriptor, by default None
        
        test_data_name : str
            Identifier of the test dataset to generate the necessary filepaths. By default ""

        Returns
        -------
        str
            Textual representation of the training parameters utilized for fine-tuning
        """
        checkpoint_name = self.model.name_or_path.replace("/","_")

        fstring = f"{checkpoint_name}_{self._train_data_name}_{test_data_name}epochs_{self.num_train_epochs}_batch_{self.batch_size}_gradient_accumulation_steps_{self.gradient_accumulation_steps}"
        return fstring if not add else f'{fstring}_{add}'

    def compute_loss(self, logits:torch.Tensor, labels:torch.Tensor, sum_idx:torch.Tensor) -> float:
        """Compute loss over the logits w.r.t. truth labels, considering to that end the subset of
        scores yielded for reference summaries.

        Parameters
        ----------
        logits : torch.Tensor
            Raw, unnormalized scores outputted by the last layer of the model.
        labels : torch.Tensor
            Collection of labels (tokenized actual summaries)
        sum_idx : torch.Tensor
            Per sample article/summary separator index

        Returns
        -------
        float
            Computed loss
        """
        # In order to fine-tune a GPT-2 architecture for summarization, both the article and its
        # actual summary must be fed altogether into the model. However, in order to compute the
        # loss, only the logits for the positions of the input corresponding to the summary shall
        # be considered. The per input sample appropriate logits can be yielded by utilizing the
        # article/summary separator (<|sep|> index).

        # For batch sizes larger than 1
        if self.batch_size > 1:
            # and because the index where separator special token resides may vary for each input
            # sample, we obtain the per sample logit slice that considers the scores on the summary
            # positions. 
            shift_logits = torch.cat([s[idx:-1, :] for s,idx in zip(logits, sum_idx)])
            shift_labels = torch.cat([s[idx+1:] for s,idx in zip(labels, sum_idx)])
            loss = self.loss_func(shift_logits, shift_labels)
        
        # In the case of online learning, this process can be simplified:
        # This snippet of code is largely based on lines [50,53] of the training script by Rohit Kumar Singh:
        # https://github.com/SKRohit/Generating_Text_Summary_With_GPT2/blob/master/train_gpt2_summarizer.py
        else:   
            idx = sum_idx.item() # index of separator token
            # The contiguous method ensures tensors are contiguous in memory, whereby indices are 
            # allocated in memory in order, i.e., they are stored canonically.
            shift_logits = logits[..., idx:-1, :].contiguous()
            shift_labels = labels[..., idx+1:].contiguous()

            # View avoids explicit data copy, thus allows us to do fast and memory efficient 
            # reshaping, slicing and element-wise operations. In this case, each element of the 
            # first dimension contains the logits for each sample of the input. This dimension is
            # squeezed into the following, providing a unified view of all samples to compute the loss
            # e.g., logits with shape (X,Y,Z) become (XY, Z). Similarly, labels are flattened (shaped as
            # a vector).
            loss = self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # Return the computed loss
        return loss

    def save_trained_model(self, 
                           model_file:Union[str,Path]=None, 
                           config_file:Union[str,Path]=None
                           ) -> None:
        """ Dump trained model (in `bin` format) and the configuration parameters (in `json` format)
        of the GPT2 model.

        Parameters
        ----------
        model_file : Union[str,Path], optional
            Custom ouput model filepath, if not specified, it is dumped to `self._model_file`,
            by default None
        config_file : Union[str,Path], optional
            Custom ouput configuration filepath, if not specified, it is dumped to `self._config_file`,
            by default None
        """
        print('Saving trained model...')
        # Unless specified, model and config files are named after the training named parameters, and
        # dumped to the ouput directory, i.e., `self.ouput_dir`
        model_file = model_file or self._model_file
        config_file = config_file or self._config_file
        
        # Save model learned parameters
        torch.save(self.model.state_dict(), model_file)
        # Save model configuration to a `json` file.
        self.model.config.to_json_file(config_file)

    def _save_train_stats(self, training_stats:list, precision=4) -> pd.DataFrame:
        """Save the statistics generated throughout the training process.

        Parameters
        ----------
        training_stats : list
            Collection of per-epoch statistics
        precision : int, optional
            Number of decimal places to use for floating-point valued fields, by 
            default 4

        Returns
        -------
        pd.DataFrame
            Statistics generated throughout the training process arranged in a DataFrame
        """
        # Display floats with four decimal places.
        pd.set_option('precision', precision)

        # Create a DataFrame from our training statistics.
        # Use the 'epoch' as the row index.
        df_stats = pd.DataFrame(data=training_stats).set_index('epoch')
        
        # Dump stats to `output_dir`
        df_stats.to_csv(Path(self.output_dir, f"train_stats_{self.attrs_to_str()}.csv"))

        return df_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="Parent directory containing at least the training and validation datasets to fine tune the model. The data should be formatted in such way that it can be processed by a `GPT2SumDataset` object. Refer to the `prepare_data.py` script for further information")
    parser.add_argument("--model", default='base', choices=["base", "large"], help="Type of BSC GPT2 architecture")
    parser.add_argument("--batch_size", type=int, required=True, help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps",default=32, type=int, required=False, help="Accumulated gradients run integer K small batches of size N before doing a backward pass.")
    parser.add_argument("--max_grad_norm",default=1.0, type=float, help="Max norm of the gradients")
    parser.add_argument("--lr",default=5e-5, type=float, required=False, help="Initial learning rate")
    parser.add_argument("--n_gpu",default=1, type=int, required=False, help="Number of GPUs available")
    parser.add_argument("--num_workers",default=1, type=int, required=False, help="Number of workers (CPUs) available")
    parser.add_argument("--device",default="cuda", choices=["cuda", "cpu"], help="torch.device object representing the device on which a torch.Tensor is or will be allocated.")
    parser.add_argument("--do_eval", action="store_true", help="Assess performance on test set (located as a subdirectory of `root_dir` and named after \"test\") once the model has been trained.")
    parser.add_argument("-o", "--output_dir",default='./output', type=str, required=True, help="Path to save the trained model and the evaluation results")
    parser.add_argument("--seed",default=42, type=int, required=False, help="Initialization state of a pseudo-random number generator to grant reproducibility of the experiments")

    parser.add_argument("--test_data_dir", required=True, type=str, help="Parent directory containing the test dataset.") 
    parser.add_argument("--max_length",default=60, type=int, help="Max summary length")
    parser.add_argument("--temperature",default=1.0, type=float, required=False, help="control the randomness of predictions by scaling the logits before applying softmax")
    parser.add_argument("--top_k",default=50, type=int, required=False, help="keep only top k tokens with highest probability (top-k filtering)")
    parser.add_argument("--top_p",default=0.9, type=float, required=False, help="keep the top tokens with cumulative probability >= top_p (nucleus filtering)")
    
    args = parser.parse_args()

    checkpoint_name = f"PlanTL-GOB-ES/gpt2-{args.model}-bne"

    bsc_summarizer = TrainGPT2Summarizer(
        checkpoint_name=checkpoint_name,
        data_dir=args.root_dir,
        batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        n_gpu=args.n_gpu,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed
    )

    start = time.time()
    print(f'Started training model: {checkpoint_name}{bsc_summarizer.attrs_to_str()}')
    history = bsc_summarizer.train()
    print('total time: ', (time.time()-start)/60, " minutes", end='\n\n')
    bsc_summarizer.save_trained_model()

    if args.do_eval:
        test_data_name = Path(args.test_data_dir).name
        test_data_name = "all_" if "all" in test_data_name else test_data_name + "_"

        generated = bsc_summarizer.generate_summaries_from_dataset(
            args.test_data_dir,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )

        test_summaries_filename = Path("summaries", "test_summaries", f"test_summaries_{test_data_name[:-1]}.txt")
        with open(test_summaries_filename, 'r', encoding='utf-8') as f:
            reference = [line.replace('\n', '') for line in f.readlines()]

        
        rouge_results_filepath = Path(Path(args.test_data_dir).parent, 'inference',f'rouge_{bsc_summarizer.attrs_to_str(add=f"temperature_{int(args.temperature)}_topk_{args.top_k}_topp_{args.top_p}_max_length_{args.max_length}", test_data_name=test_data_name)}.csv')

        compute_rouge_score(reference, generated, rouge_results_filepath)