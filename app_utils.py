from gpt2_summarizer_inference import InferenceGPT2Summarizer
from pathlib import Path

def load_dependencies(checkpoint=None):
    args = get_params_from_model_id(checkpoint) if checkpoint else get_params_default()
    print(f'Fine-tuned model args: {args}')
    bsc_summarizer_test = InferenceGPT2Summarizer(
        **args,
        num_workers=1,
        device="cpu",
        output_dir="output"
    )
    return bsc_summarizer_test

def get_params_from_model_id(model_id):
    checkpoint_name = '/'.join(model_id.split('_')[:2])
    batch_size = get_field_from_model_id(model_id, 'batch')
    num_train_epochs = get_field_from_model_id(model_id, 'epochs')
    gradient_accumulation_steps = get_field_from_model_id(model_id, 'gradient_accumulation_steps')
    train_data_name=get_dataname_from_model_id(model_id)
    return dict(
        checkpoint_name=checkpoint_name,
        train_data_name=train_data_name,
        batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps
    )

def get_params_default():
    return dict(
        checkpoint_name="PlanTL-GOB-ES/gpt2-base-bne",
        train_data_name="all",
        batch_size=8,
        num_train_epochs=3,
        gradient_accumulation_steps=32
    )

def get_field_from_model_id(model_id, field):
    split_ = model_id.split(f'{field}_')
    return int(split_[1].split('_')[0]) if '_' in split_[-1] else int(split_[1])

def get_dataname_from_model_id(model_id):
    if 'all' in model_id:
        return 'all'
    else:
        return f'tokenized_{get_field_from_model_id(model_id, "tokenized")}'

def path_to_rouge(model_id):
    DIR_INFERENCE=Path("summaries", "inference")
    if Path(DIR_INFERENCE, f'rouge_{model_id}_temperature_1_topk_10_topp_0.5.csv').exists():
        return Path(DIR_INFERENCE, f'rouge_{model_id}_temperature_1_topk_10_topp_0.5.csv')
    if 'all' in model_id:
        print(model_id.replace("all_", "all_all_"))
        print(model_id.replace("all_", "all_tokenized_512_"))
        a = list(Path(DIR_INFERENCE).glob(f'rouge_{model_id.replace("all_", "all_all_")}_*.csv'))
        b = list(Path(DIR_INFERENCE).glob(f'rouge_{model_id.replace("all_", "all_tokenized_512_")}_*.csv'))
        c = a + b
        return c[0]
    else:
        print(model_id.replace("tokenized_512_", "tokenized_512_all_"))
        print(model_id.replace("tokenized_512_", "tokenized_512_tokenized_512_"))
        a = list(Path(DIR_INFERENCE).glob(f'rouge_{model_id.replace("tokenized_512_", "tokenized_512_all_")}_*.csv'))
        b = list(Path(DIR_INFERENCE).glob(f'rouge_{model_id.replace("tokenized_512_", "tokenized_512_tokenized_512_")}_*.csv'))
        c = a + b
        return c[0]

def get_model_paths():
    DIR_OUTPUT=Path("output")
    DIR_INFERENCE=Path("summaries", "inference")
    path_to_train_stats = lambda x: Path(DIR_OUTPUT, f'train_stats_{x}.csv')
    path_to_model = lambda x: Path(DIR_OUTPUT, f'model_{x}.bin')
    path_to_config = lambda x: Path(DIR_OUTPUT, f'config_{x}.json')
    # path_to_rouge = lambda x: Path(DIR_INFERENCE, f'rouge_{x}_temperature_1_topk_10_topp_0.5.csv')
    # path_to_rouge = lambda x: list(Path(DIR_INFERENCE).glob(f'rouge_{x}_*.csv'))[0]

    model_files = DIR_OUTPUT.glob('model*.bin')
    model_names = [str(model.name).replace('model_', '').replace('.bin', '') for model in model_files]

    return {model: {'model': path_to_model(model), 'config': path_to_config(model), 'train_stats': path_to_train_stats(model), 'rouge':path_to_rouge(model)} for model in model_names}
