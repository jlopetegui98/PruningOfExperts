import torch
import itertools
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer, default_data_collator
from transformers.testing_utils import CaptureLogger
import transformers
import gzip
import json
import urllib.request

DATASETS = {
    'c4': lambda: load_dataset('json', data_files={'train': 'drive/MyDrive/LLMs_MVA/Final_Project/c4-train.00000-of-01024.json'}),
    'math': lambda: load_dataset('json', data_files={'train': 'drive/MyDrive/LLMs_MVA/Final_Project/math_pretrain_style.json'}),
}

class CacheDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.alphas = []
        self.Xs = []
        self.Zs = []
        self.prepared = False

    def __len__(self):
        if not self.prepared:
            self.prepare_for_loader()
        return len(self.alphas)

    def __getitem__(self, index):
        if not self.prepared:
            self.prepare_for_loader()
        if isinstance(index, list):
            return [(self.alphas[idx], self.Xs[idx], self.Zs[idx]) for idx in index]
        elif isinstance(index, int):
            return self.alphas[index], self.Xs[index], self.Zs[index]

    def append(self, alpha=None, X=None, Z=None):
        if alpha is not None:
            self.alphas.append(alpha.detach().to('cpu', non_blocking=True))
        if X is not None:
            self.Xs.append(X.detach().to('cpu', non_blocking=True))
        if Z is not None:
            self.Zs.append(Z.detach().to('cpu', non_blocking=True))
        self.prepared = False

    def prepare_for_loader(self):
        if self.prepared:
            return
        self.prepared = True
        self.alphas = torch.concat(self.alphas)
        self.Xs = torch.concat(self.Xs)
        self.Zs = torch.concat(self.Zs)
        assert len(self.Xs) == len(self.Zs)

def build_calib_loader_mixtral(dataset: str, tokenizer, max_block_size: int, n_blocks_for_stat: int, batch_size: int, num_workers: int, seed: int = 42):
    all_set = DATASETS[dataset]()

    block_size = tokenizer.model_max_length
    if block_size > max_block_size:
        print(
            "The chosen tokenizer supports a `model_max_length` that is longer than the default `max_block_size` value"
            f" of {max_block_size}. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
            " override this default with `--max_block_size xxx`."
        )
        block_size = max_block_size

    if n_blocks_for_stat > 0:  # Random choose `n_blocks_for_stat` blocks
        calib_set = all_set['train'].shuffle(seed=seed).select(
            range(min(n_blocks_for_stat * 16, len(all_set['train']))))
    else:   # Use the whole set
        print('n_blocks_for_stat <= 0, using the whole dataset.')
        calib_set = all_set['train'].shuffle(seed=seed)

    print(f'Calibration dataset: {calib_set}')
    text_column_name = "text" if "text" in calib_set.features else list(
        calib_set.features)[0]

    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output
    tokenized_calib_set = calib_set.map(
        tokenize_function,
        batched=True,
        remove_columns=list(calib_set.features),
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(itertools.chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    lm_calib_set = tokenized_calib_set.map(
        group_texts,
        batched=True,
    )

    if n_blocks_for_stat > 0:
        assert len(lm_calib_set) > n_blocks_for_stat
        lm_calib_set = lm_calib_set.select(range(n_blocks_for_stat))

    calib_loader = torch.utils.data.DataLoader(
        lm_calib_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=default_data_collator
    )

    return calib_loader


def build_calib_loader_deepseek(dataset_name, tokenizer, max_block_size=2048, n_blocks=128, batch_size=4, num_workers=8, seed=42):

    if dataset_name == 'c4':
        url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
        local_file = "c4-train.00000-of-01024.json.gz"

        # Download the file if it doesn't exist
        if not os.path.exists(local_file):
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, local_file)

        # Read and decompress the file
        print("Reading and decompressing the dataset...")
        with gzip.open(local_file, 'rt', encoding='utf-8') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= n_blocks * 16:
                    break
                try:
                    item = json.loads(line)
                    lines.append(item)
                except json.JSONDecodeError:
                    continue

        all_set = {'train': Dataset.from_list(lines)}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    block_size = tokenizer.model_max_length
    if block_size > max_block_size:
        print(f"Using max_block_size={max_block_size} instead of tokenizer.model_max_length={tokenizer.model_max_length}")
        block_size = max_block_size

    if n_blocks > 0:
        calib_set = all_set['train'].shuffle(seed=seed).select(
            range(min(n_blocks * 16, len(all_set['train']))))
    else:
        print('n_blocks <= 0, using the whole dataset.')
        calib_set = all_set['train'].shuffle(seed=seed)

    print(f'Calibration dataset: {calib_set}')

    text_column_name = "text" if "text" in calib_set.features else list(calib_set.features)[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_calib_set = calib_set.map(
        tokenize_function,
        batched=True,
        remove_columns=list(calib_set.features),
    )

    def group_texts(examples):
        concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_calib_set = tokenized_calib_set.map(
        group_texts,
        batched=True,
    )

    if n_blocks > 0:
        assert len(lm_calib_set) > n_blocks
        lm_calib_set = lm_calib_set.select(range(n_blocks))

    calib_loader = torch.utils.data.DataLoader(
        lm_calib_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=default_data_collator
    )

    return calib_loader
