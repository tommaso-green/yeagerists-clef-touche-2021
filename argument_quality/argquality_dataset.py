import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from functools import partial
from datasets import load_dataset, load_from_disk
from transformers import AutoModel, AutoTokenizer


def train_val_test_split(dataset, split_dict):
    data = dataset['train']
    train_test_split = data.train_test_split(train_size=split_dict['train'] + split_dict['val'],
                                             test_size=split_dict['test'])
    train_val_split = data.train_test_split(train_size=split_dict['train'], test_size=split_dict['val'])
    dataset['train'] = train_val_split['train']
    dataset['val'] = train_val_split['test']
    dataset['test'] = train_test_split['test']
    return dataset


def prepare_batch(data_batch, model_tokenizer):
    padded_arguments = model_tokenizer.pad(data_batch, return_tensors='pt')
    quality_scores = padded_arguments['Combined Quality']
    del padded_arguments['Combined Quality']
    return padded_arguments, quality_scores.unsqueeze(1)


class ArgQualityDataset(pl.LightningDataModule):

    def __init__(self, batch_size, tokenizer_name, split_dict, csv_path):
        super().__init__()

        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.split_dict = split_dict
        self.csv_path = csv_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def tokenize(self, example):
        return self.tokenizer(example['Premise'], padding=False, truncation=True) #, max_length=300)

    def prepare_data(self):
        full_dataset = load_dataset('csv', data_files=self.csv_path)
        dataset_split = train_val_test_split(full_dataset, self.split_dict)

        dataset_split = dataset_split.map(self.tokenize)
        dataset_split.save_to_disk("dataset_dir")

    def setup(self, stage=None):
        dataset_split = load_from_disk("dataset_dir")
        cols_to_keep = [x for x in ['input_ids', 'token_type_ids', 'attention_mask', 'Combined Quality'] if x in dataset_split['train'].column_names]

        self.train_dataset = dataset_split['train']
        self.train_dataset.set_format(type='torch', columns=cols_to_keep)

        self.val_dataset = dataset_split['val']
        self.val_dataset.set_format(type='torch', columns=cols_to_keep)

        self.test_dataset = dataset_split['test']
        self.test_dataset.set_format(type='torch', columns=cols_to_keep)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                           collate_fn=partial(prepare_batch, model_tokenizer=self.tokenizer))

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                           collate_fn=partial(prepare_batch, model_tokenizer=self.tokenizer))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                                           collate_fn=partial(prepare_batch, model_tokenizer=self.tokenizer))

    @staticmethod
    def add_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ArgQualityDataset")
        parser.add_argument('-b', '--batch_size', type=int, default=8)
        parser.add_argument('-p', '--csv_path', type=str, default='../webis-argquality20-full.csv')
        return parent_parser

def test():
    split_dict = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # download checkpoint for Bert Tokenizer
    dm = ArgQualityDataset(5, tokenizer, split_dict, 'webis-argquality20-full.csv')
    dm.prepare_data()
    dm.setup()
    tr_lod = dm.train_dataloader()
    b = next(iter(tr_lod))
    print(b)
    for k in b.keys():
        print(b[k].shape)
