import os
import copy
import random
import math
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AlbertTokenizer, DataCollatorWithPadding

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler


class DataManager(object):

    def __init__(self, config):
        self.config = config
        self.init_gpu_config()

    def init_gpu_config(self):
        """
        初始化GPU并行配置
        """
        print('loading GPU config ...')
        if self.config.mode == 'train' and torch.cuda.device_count() > 1:
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method=self.config.init_method,
                                                 rank=0,
                                                 world_size=self.config.world_size)
            torch.distributed.barrier()

    def get_dataset(self, mode='train', sampler=True):
        """
        获取数据集
        """
        # 读取tokenizer分词模型
        tokenizer = AlbertTokenizer.from_pretrained(self.config.initial_pretrain_tokenizer)

        if mode == 'train':
            train_dataloader = self.data_process('train.txt', tokenizer)
            return train_dataloader
        elif mode == 'dev':
            eval_dataloader = self.data_process('dev.txt', tokenizer)
            return eval_dataloader
        else:
            test_dataloader = self.data_process('test.txt', tokenizer, sampler=sampler)
            return test_dataloader
    def open_file(self, path):
        """读文件"""
        text = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip()
                text.append(line)
        return text
    def data_process(self, file_name, tokenizer, sampler=True):
        """
        数据转换
        """
        # 获取数据
        text = self.open_file(self.config.path_datasets + file_name)
        dataset = pd.DataFrame({'src': text, 'labels': text})
        # dataframe to datasets
        raw_datasets = Dataset.from_pandas(dataset)
        # tokenizer.
        tokenized_datasets = raw_datasets.map(lambda x: self.tokenize_function(x, tokenizer), batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        tokenized_datasets = tokenized_datasets.remove_columns(["src"])
        tokenized_datasets.set_format("torch")

        if sampler:
            sampler = RandomSampler(tokenized_datasets) if not torch.cuda.device_count() > 1 else DistributedSampler(
                tokenized_datasets)
        else:
            sampler = None
        dataloader = DataLoader(tokenized_datasets, sampler=sampler, batch_size=self.config.batch_size)
        return dataloader

    def tokenize_function(self, example, tokenizer):
        """
        数据转换
        """
        token = tokenizer(example["src"], truncation=True, max_length=self.config.sen_max_length, padding='max_length')
        label = copy.deepcopy(token.data['input_ids'])
        token.data['labels'] = label
        token_mask = tokenizer.mask_token
        token_pad = tokenizer.pad_token
        token_cls = tokenizer.cls_token
        token_sep = tokenizer.sep_token
        ids_mask = tokenizer.convert_tokens_to_ids(token_mask)
        token_ex = [token_mask, token_pad, token_cls, token_sep]
        ids_ex = [tokenizer.convert_tokens_to_ids(x) for x in token_ex]
        vocab = tokenizer.get_vocab()
        vocab_int2str = {v: k for k, v in vocab.items()}
        if self.config.whole_words_mask:
            mask_token = [self.op_mask_wwm(line, ids_mask, ids_ex, vocab_int2str) for line in token.data['input_ids']]
        else:
            mask_token = [[self.op_mask(x, ids_mask, ids_ex, vocab) for i, x in enumerate(line)] for line in
                          token.data['input_ids']]
        mask_token_len = len(set([len(x) for x in mask_token]))
        assert mask_token_len == 1, 'length of mask_token not equal.'
        flag_input_label = [1 if len(x) == len(y) else 0 for x, y in zip(mask_token, label)]
        assert sum(flag_input_label) == len(mask_token), 'the length between input and label not equal.'
        token.data['input_ids'] = mask_token
        return token

    def op_mask(self, token, ids_mask, ids_ex, vocab):
        """
        ALBERT的原始mask机制。
            （1）80%的概率，保留原词不变
            （2）10%的概率，使用字符'[MASK]'，替换当前token。
            （3）10%的概率，使用词表随机抽取的token，替换当前token。
        """
        if token in ids_ex:
            return token
        if random.random() <= 0.10:
            x = random.random()
            if x <= 0.80:
                token = ids_mask
            if x > 0.80 and x <= 0.9:
                while True:
                    token = random.randint(0, len(vocab) - 1)
                    if token not in ids_ex:
                        break
        return token

    def op_mask_wwm(self, tokens, ids_mask, ids_ex, vocab_int2str):
        if len(tokens) <= 5:
            return tokens

        line = copy.deepcopy(tokens)
        for i, token in enumerate(tokens):
            if token is None:
                continue  # Skip empty tokens

            if token in ids_ex:
                line[i] = token
                continue

            if random.random() <= 0.10:
                x = random.random()
                if x <= 0.80:
                    token_str = vocab_int2str[token]
                    if '##' in token_str:
                        line[i] = ids_mask
                        curr_i = i - 1
                        flag = True
                        while flag:
                            token_index = tokens[curr_i]
                            token_index_str = vocab_int2str[token_index]
                            if '##' not in token_index_str:
                                flag = False
                            line[curr_i] = ids_mask
                            curr_i -= 1
                        curr_i = i + 1
                        flag = True
                        while flag:
                            token_index = tokens[curr_i]
                            token_index_str = vocab_int2str[token_index]
                            if '##' not in token_index_str:
                                flag = False
                            else:
                                line[curr_i] = ids_mask
                            curr_i += 1
                    else:
                        line[i] = ids_mask
                        curr_i = i + 1
                        flag = True
                        while flag:
                            token_index = tokens[curr_i]
                            token_index_str = vocab_int2str[token_index]
                            if '##' not in token_index_str:
                                flag = False
                            else:
                                line[curr_i] = ids_mask
                            curr_i += 1
                if x > 0.80 and x <= 0.9:
                    while True:
                        token = random.randint(0, len(vocab_int2str) - 1)
                        if token not in ids_ex:
                            break
        return line