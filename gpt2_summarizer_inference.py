import argparse
from pathlib import Path
import time
from transformers import GPT2Config, GPT2LMHeadModel
import torch

from typing import Union

from gpt2_summarizer import GPT2Summarizer 
from utils import compute_rouge_score


class InferenceGPT2Summarizer(GPT2Summarizer):

    def __init__(self,
                 checkpoint_name:str,
                 train_data_name:str,
                 batch_size:int,
                 num_train_epochs:int,
                 gradient_accumulation_steps:int,
                 num_workers:int,
                 device:torch.device,
                 output_dir:Union[str, Path]
                ) -> None:
        """Constructor of a `InferenceGPT2Summarizer` instance, which provides all necessary functionality
        to allow the use of a fine-tuned GPT2-like architecture for abstractive summarization at inference
        i.e., to generate summaries on unseen data, both in an as-is basis (see `self.generate_sample`) or
        provided a `GPT2SumDataset` (see `self.generate_summaries_from_dataset`).

        Parameters
        ----------
        checkpoint_name : str
            Model id of a pretrained HuggingFace Transformer hosted inside a model repo on 
            huggingface.co
        train_data_name : str
            Identifier of the training dataset on which the model has been trained.
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
       # _train_data_name is used to generate output filenames automatically
        self._train_data_name = train_data_name
        # Store checkpoint name
        self._checkpoint_name = checkpoint_name
        # Load all necessary info. from disk and setup the fine-tuned model
        self.setup_model()       

        # Set the model in evaluation mode (no change of learnable parameters)
        self.model.eval()
        # Send the model to the appropriate device
        self.model.to(self._device)

    @property
    def config(self) -> GPT2Config:
        """GPT2 configuration object. It is used to instantiate a GPT-2 model 
        according to the specified arguments, defining the model architecture.

        Returns
        -------
        GPT2Config
            GPT2 configuration object
        """
        try:
            return self._config
        except AttributeError:
            print(f"Loading GPT2 configuration from json file {self._config_file}")
            self._config = GPT2Config.from_json_file(self._config_file)
            return self._config

    @property
    def model(self) -> GPT2LMHeadModel:
        """GPT2 model fine tuned for abstractive summarization

        Returns
        -------
        GPT2LMHeadModel
            GPT2 model fine tuned for abstractive summarization
        """
        try:
            return self._model
        except AttributeError:
            self.setup_model()
            return self._model

    @property
    def _config_file(self) -> Path:
        """Filepath of the configuration file (in `json` format) of the trained model. This file
        is loaded from the output directory (refer to `self.output_dir` prop.), and named after 
        the "named" parameters utilized at training (refer to `self.attrs_to_str()` module). 

        Returns
        -------
        Path
            Input filepath for the configuration file of the fine-tuned model
        """
        return Path(self.output_dir, f'config_{self.attrs_to_str()}.json')

    @property
    def _model_file(self) -> Path:
        """Filepath for the trained model binary (in `bin` format). This file is loaded from the 
        output directory (refer to `self.output_dir` prop.), and named after the "named" parameters 
        utilized at training (refer to `self.attrs_to_str()` module). 

        Returns
        -------
        Path
            Input filepath for the trained model binary
        """
        return Path(self.output_dir, f'model_{self.attrs_to_str()}.bin')

    @property
    def state_dict(self) -> dict:
        """Python dictionary object that maps each layer with learnable parameters (i.e., weights 
        and biases) to its parameter tensor. 

        Returns
        -------
        dict
            Model learnable parameters
        """
        try:
            return self._state_dict
        except AttributeError:
            if self._device == "cuda":
                self._state_dict = torch.load(self._model_file)
            else:
                self._state_dict = torch.load(self._model_file, map_location=torch.device(self._device))
            return self._state_dict

    def attrs_to_str(self, add:str=None, test_data_name="") -> str:
        """Yield a string-like representation of the training attributes to fine-tune the model,
        serving as a straightforward and effective fashion to uniquely identify the model. 
        
        Parameters
        ----------
        add : str, optional
            Substring to append at the end of the string model descriptor, by default None

        test_data_name : str
            Identifier of the test dataset to generate the necessary filepaths. By default, ""

        Returns
        -------
        str
            Textual representation of the training parameters utilized for fine-tuning
        """
        checkpoint_name = self._checkpoint_name.replace("/","_")

        fstring = f"{checkpoint_name}_{self._train_data_name}_{test_data_name}epochs_{self.num_train_epochs}_batch_{self.batch_size}_gradient_accumulation_steps_{self.gradient_accumulation_steps}"
        return fstring if not add else f'{fstring}_{add}'

    def setup_model(self, config:GPT2Config=None, state_dict:dict=None) -> None:
        """Setup fine-tuned model from the configuration object and the learned parameters.

        Parameters
        ----------
        config : GPT2Config, optional
            Custom model architecture configuration object. If not specified, the configuration
            used is that of `self.config`, which instantiates a `GPT2Config` object from 
            `self._config_file` json configuration file
        state_dict : dict, optional
            Custom `state_dict` object. If not specified, the learned parameters used are those
            of `self.state_dict`, which are deserialized from `self._model_file`
        """
        # Load default state_dict in absence of custom learned parameters dict
        state_dict = state_dict or self.state_dict
        # Load default model architecture in absence of custom configuration object
        config = config or self.config
        print("Loading GPT2 fine-tuned...")
        # Get the model architecture from the configuration object
        self._model = GPT2LMHeadModel(config)
        # Load the learned parameters
        self._model.load_state_dict(state_dict)

# execution example
# python gpt2_summarizer_inference.py --root_dir summaries/tokenized_512 --model base --num_train_epochs 10 --batch_size 1 --num_workers 1 --device cpu --model_dir output/
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", required=True, type=str, help="Parent directory containing the training dataset on which the model has been trained.")
    parser.add_argument("--test_data_dir", required=True, type=str, help="Parent directory containing the test dataset.") 
    parser.add_argument("--model", default='base', choices=["base", "large"], help="Type of BSC GPT2 architecture")
    parser.add_argument("--batch_size",default=1, type=int, required=True, help="batch_size")
    parser.add_argument("--num_train_epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps",default=32, type=int, required=False, help="Accumulated gradients run integer K small batches of size N before doing a backward pass.")
    parser.add_argument("--max_length",default=60, type=int, help="Max summary length")
    parser.add_argument("--num_workers",default=1, type=int, required=False, help="Number of workers (CPUs) available")
    parser.add_argument("--temperature",default=1.0, type=float, required=False, help="Introduce randomness of the predictions by scaling the model logits before applying softmax")
    parser.add_argument("--top_k",default=10, type=int, required=False, help="Keep only top k tokens with highest probability (top-k filtering)")
    parser.add_argument("--top_p",default=0.5, type=float, required=False, help="Keep the top tokens with cumulative probability >= top_p (nucleus filtering)")
    parser.add_argument("--device",default=torch.device('cuda'), required=False, help="torch.device object")
    parser.add_argument("-o", "--output_dir",default='./output', type=str, required=True, help="path to save the trained model and evaluation results")


    args = parser.parse_args()

    checkpoint_name = f"PlanTL-GOB-ES/gpt2-{args.model}-bne"

    train_data_name = Path(args.train_data_dir).name
    train_data_name = "all"  if "all" in train_data_name else train_data_name

    test_data_name = Path(args.test_data_dir).name
    test_data_name = "all_" if "all" in test_data_name else test_data_name + "_"

    bsc_summarizer_test = InferenceGPT2Summarizer(
        checkpoint_name=checkpoint_name,
        train_data_name=train_data_name,
        batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=args.output_dir
    )
    print('attrs_to_str ', bsc_summarizer_test.attrs_to_str(add=f"temperature_{int(args.temperature)}_topk_{args.top_k}_topp_{args.top_p}_max_length_{args.max_length}", test_data_name=test_data_name))

    start = time.time()
    generated = bsc_summarizer_test.generate_summaries_from_dataset(
        args.test_data_dir,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    test_summaries_filename = Path("summaries", "test_summaries", f"test_summaries_{test_data_name[:-1]}.txt")
    with open(test_summaries_filename, 'r', encoding='utf-8') as f:
        reference = [line.replace('\n', '') for line in f.readlines()]

    # Filepath where info about ROUGE metrics will be stored
    rouge_results_filepath = Path(Path(args.test_data_dir).parent, 'inference',f'rouge_{bsc_summarizer_test.attrs_to_str(add=f"temperature_{int(args.temperature)}_topk_{args.top_k}_topp_{args.top_p}_max_length_{args.max_length}", test_data_name=test_data_name)}.csv')

    compute_rouge_score(reference, generated, rouge_results_filepath)

    print('total time: ', (time.time()-start)/60, " minutes", end='\n\n')

