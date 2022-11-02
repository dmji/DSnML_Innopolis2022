import logging
from pathlib import Path
from typing import List, Mapping, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os

exit(-1)

import yaml
from catalyst.utils import set_global_seed, prepare_cudnn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from catalyst.callbacks.metrics.accuracy import AccuracyCallback
from catalyst.dl import (
    CheckpointCallback,
    OptimizerCallback,
    SchedulerCallback,
    SupervisedRunner,
)

project_root: Path = Path("").parent.parent


SEED = 17
PATH_TO_LOG_FOLDER = Path('logdir')

 
# ## Data Load


class TextClassificationDataset(Dataset):
    """
    Wrapper around Torch Dataset to perform text classification
    """

    def __init__(
        self,
        texts: List[str],
        questions: List[str],
        labels: List[str] = None,
        label_dict: Mapping[str, int] = None,
        max_seq_length: int = None,
        model_name: str = None,
    ):
        """
        Args:
            texts (List[str]): a list with texts to classify or to train the
                classifier on
            labels List[str]: a list with classification labels (optional)
            label_dict (dict): a dictionary mapping class names to class ids,
                to be passed to the validation data (optional)
            max_seq_length (int): maximal sequence length in tokens,
                texts will be stripped to this length
            model_name (str): transformer model name, needed to perform
                appropriate tokenization

        """

        self.texts = texts
        self.questions = questions
        self.labels = labels
        self.label_dict = label_dict
        self.max_seq_length = max_seq_length

        if self.label_dict is None and labels is not None:
            # {'class1': 0, 'class2': 1, 'class3': 2, ...}
            # using this instead of `sklearn.preprocessing.LabelEncoder`
            # no easily handle unknown target values
            self.label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # suppresses tokenizer warnings
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.FATAL)

        # special tokens for transformers
        # in the simplest case a [CLS] token is added in the beginning
        # and [SEP] token is added in the end of a piece of text
        # [CLS] <indexes text tokens> [SEP] .. <[PAD]>
        self.sep_label = self.tokenizer.special_tokens_map['sep_token']
        self.sep_vid = self.tokenizer.vocab[self.sep_label]
        self.cls_label = self.tokenizer.special_tokens_map['cls_token']
        self.cls_vid = self.tokenizer.vocab[self.cls_label]
        self.pad_label = self.tokenizer.special_tokens_map['pad_token']
        self.pad_vid = self.tokenizer.vocab[self.pad_label]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.texts)

    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:
        """Gets element of the dataset

        Args:
            index (int): index of the element in the dataset
        Returns:
            Single element by index
        """
        # encoding the text
        x = [self.texts[index], self.questions[index]]

        # a dictionary with `input_ids` and `attention_mask` as keys
        output_dict = self.tokenizer.encode_plus(
            text=x,
            ##text_pair=text_pair, 
            ##text_target=text_target, 
            ##text_pair_target=text_pair_target, 
            add_special_tokens=True,
            #  Pad to a maximum length specified with the argument max_length
            #  or to the maximum acceptable input length for the model if that argument is not provided.
            padding="max_length", 
            # Truncate to a maximum length specified with the argument max_length 
            # or to the maximum acceptable input length for the model if that argument is not provided. 
            # This will truncate token by token, 
            # removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided.
            truncation=True,
            # Controls the maximum length to use by one of the truncation/padding parameters.
            max_length=self.max_seq_length, 
            # return pytorch tensor
            return_tensors="pt", 
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        # for Catalyst, there needs to be a key called features
        output_dict["features"] = output_dict["input_ids"].squeeze(0)
        del output_dict["input_ids"]

        output_dict["token_type_ids"] = output_dict["token_type_ids"].squeeze(0)

        # encoding target
        if self.labels is not None:
            y = self.labels[index]
            y_encoded = torch.Tensor([self.label_dict.get(y, -1)]).long().squeeze(0)
            output_dict["targets"] = y_encoded
            
        return output_dict


def read_data(params: dict) -> Tuple[dict, dict]:
    """
    A custom function that reads data from CSV files, creates PyTorch datasets and
    data loaders. The output is provided to be easily used with Catalyst

    :param params: a dictionary read from the config.yml file
    :return: a tuple with 2 dictionaries
    """

    batch_size              = params["training"]["batch_size"]
    seed                    = SEED
    max_seq_length          = params["model"]["max_seq_length"]
    model_name_path_or_url  = params["model"]["model_name"]
    dataset_folder          = params["data"]["path_to_data"]
    context_column          = params["data"]["text_field_name"]
    question_column         = params["data"]["quest_field_name"]
    label_column            = params["data"]["label_field_name"]
    train_filename          = params["data"]["train_filename"]
    validation_filename     = params["data"]["validation_filename"]
    test_filename           = params["data"]["test_filename"]

    # reading CSV files to Pandas dataframes
    train_df = pd.read_json(Path(dataset_folder) / train_filename, lines=True)
    valid_df = pd.read_json(Path(dataset_folder) / validation_filename, lines=True)
    test_df  = pd.read_json(Path(dataset_folder) / test_filename, lines=True)

    # делаем маппинг из True/False в INT
    label_dict = dict()
    for i, v in enumerate(np.unique(valid_df[label_column].values)):
        label_dict[v] = i

    # creating PyTorch Datasets
    train_dataset = TextClassificationDataset(
        texts           = train_df[context_column].values.tolist(),
        questions       = train_df[question_column].values.tolist(),
        labels          = train_df[label_column].values,
        label_dict      = label_dict,
        max_seq_length  = max_seq_length,
        model_name      = model_name_path_or_url
    )

    valid_dataset = TextClassificationDataset(
        texts           = valid_df[context_column].values.tolist(),
        questions       = valid_df[question_column].values.tolist(),
        labels          = valid_df[label_column].values,
        label_dict      = label_dict,
        max_seq_length  = max_seq_length,
        model_name      = model_name_path_or_url
    )

    test_dataset = TextClassificationDataset(
        texts           = test_df[context_column].values.tolist(),
        questions       = test_df[question_column].values.tolist(),
        max_seq_length  = max_seq_length,
        model_name      = model_name_path_or_url,
    )

    set_global_seed(seed)

    # creating PyTorch data loaders and placing them in dictionaries (for Catalyst)
    train_val_loaders = {
        "train": DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
        ),
        "valid": DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
        ),
    }

    test_loaders = {
        "test": DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    }

    del train_df, valid_df, test_df, train_dataset, valid_dataset, test_dataset
    del test_filename, validation_filename, train_filename, seed, model_name_path_or_url, max_seq_length, label_column, question_column, context_column, dataset_folder, batch_size
    return train_val_loaders, test_loaders

 
# ## Model


class BertForSequenceClassification(nn.Module):
    """
    Simplified version of the same class by HuggingFace.
    See transformers/modeling_distilbert.py in the transformers repository.
    """

    def __init__(
        self, 
        pretrained_model_name: str, 
        num_classes: int = None, 
        dropout: float = 0.3
    ):
        """
        Args:
            pretrained_model_name (str): HuggingFace model name.
                See transformers/modeling_auto.py
            num_classes (int): the number of class labels
                in the classification task
        """
        super().__init__()

        config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=num_classes)

        self.model = AutoModel.from_pretrained(pretrained_model_name, config=config)
        self.dropout = nn.Dropout(dropout)

        if 1: #default out-of-box gateway
            self.classifier = nn.Linear(config.hidden_size, num_classes)
        else:
            self.classifier = nn.Linear(config.hidden_size * 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, **kwargs):
        attention_mask = kwargs['attention_mask']
        features = kwargs['features']
        token_type_ids = kwargs['token_type_ids']
        head_mask = None
    #def forward(self, features, attention_mask=None, token_type_ids=None, head_mask=None):
        """Compute class probabilities for the input sequence.

        Args:
            features (torch.Tensor): ids of each token,
                size ([bs, seq_length]
            attention_mask (torch.Tensor): binary tensor, used to select
                tokens which are used to compute attention scores
                in the self-attention heads, size [bs, seq_length]
            head_mask (torch.Tensor): 1.0 in head_mask indicates that
                we keep the head, size: [num_heads]
                or [num_hidden_layers x num_heads]
        Returns:
            PyTorch Tensor with predicted class scores
        """
        assert attention_mask is not None, "attention mask is none"

        # taking BERTModel output
        # see https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel
        bert_output = self.model(
            input_ids=features, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            head_mask=head_mask)
        # we only need the hidden state here and don't need
        # transformer output, so index 0

        if 1: #default out-of-box gateway
            seq_output = bert_output[0]  # (bs, seq_len, dim)
            # mean pooling, i.e. getting average representation of all tokens
            pooled_output = seq_output.mean(axis=1)  # (bs, dim)
            pooled_output = self.dropout(pooled_output)  # (bs, dim)
            scores = self.classifier(pooled_output)  # (bs, num_classes)
        else:
            encoder_out = bert_output['last_hidden_state']
            
            pooled_output, _ = torch.max(encoder_out, 1)
            pooled_output = torch.relu(pooled_output)
            
            pooled_output_mean = torch.mean(encoder_out, 1)
            #cls = bert_output[:, 0, :]
            pooled_output = torch.cat((pooled_output, pooled_output_mean), 1)
            
            pooled_output = self.dropout(pooled_output)
            scores = self.classifier(pooled_output)
            
            #scores = (logits,) + scores[2:]  # add hidden states and attention if they are here

            #if self.model labels is not None:
            #    if self.num_labels == 1:
            #        # We are doing regression
            #        loss_fct = MSELoss()
            #        loss = loss_fct(logits.view(-1), labels.view(-1))
            #    else:
            #        loss = torch.nn.functional.binary_cross_entropy_with_logits( logits.view(-1), labels.view(-1) )
            #    outputs = (loss,) + outputs
        scores = self.softmax(scores)
        return scores

 
# ## Train

 
# Из-за того, что Cuda не умеет освобождать VRAM каждый раз приходится перезапускать ядро поэтому делать прогон серии конфигураций не имеет смысла


def getResultPath(config_key):
    return PATH_TO_LOG_FOLDER / config_key / "csv_logger" / "valid.csv"

def getConfigPaths(aL, aModels, aSeqSize):
    configs_dict = dict()
    for iL in aL:
        for iM in aModels:
            for iS in aSeqSize:
                name = f'L{iL}_M{iM}_S{iS}'
                path = str(project_root / "configs" / f"config_{name}.yml")
                if os.path.exists(path):
                    configs_dict[name] = path
    return configs_dict
configs_dict = getConfigPaths(range(0, 6), range(1, 3), [2**i for i in range(1, 10)])

# открываем файл конфига и зачитываем параметры
config_key = None
for config in configs_dict:
    if not os.path.exists(getResultPath(config)):
        config_key = config

if config_key == None:
    exit(1)

with open(configs_dict[config_key]) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# загружаем датасет
train_val_loaders, test_loaders = read_data(params)

# загружаем модель из параметров с задангным кол-вом классов
model = BertForSequenceClassification(
    pretrained_model_name=params["model"]["model_name"],
    num_classes=params["model"]["num_classes"],
)

if 1:
    param_optimizer = list(model.model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
else:
    optimizer_grouped_parameters = model.parameters()

# specify criterion for the multi-class classification task, optimizer and scheduler
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=float(params["training"]["learn_rate"]))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# сбрасываем сид рандомайзеров
set_global_seed(SEED)
prepare_cudnn(deterministic=True)

# определяем тренера, который будет оперировать с forward методом, 
# input_key - ключи, которые будут преедаваться из тензора в forward
runner = SupervisedRunner(input_key=("features", "attention_mask", "token_type_ids"))
logdir_path = PATH_TO_LOG_FOLDER / config_key
metric_key = 'accuracy01' #'loss'
# запускаем обучение
runner.train(
    # модель
    model=model,
    # функция потерь
    criterion=criterion,
    # оптимизатор
    optimizer=optimizer,
    # расписание
    scheduler=scheduler,
    # словарь из тренировочного и тестового датасета
    loaders=train_val_loaders,
    # функции обратной связи
    callbacks=[
        AccuracyCallback(num_classes=int(params["model"]["num_classes"]), input_key="logits", target_key="targets"),
        OptimizerCallback(accumulation_steps=int(params["training"]["accum_steps"]), metric_key=metric_key),
        SchedulerCallback(loader_key="valid", metric_key=metric_key),
        CheckpointCallback(logdir=logdir_path, loader_key="valid", metric_key=metric_key, minimize=True),
    ],
    # путь до папки, в которую будут сохраняться результаты, 
    # промежуточные модели: лучшая и последняя
    logdir=logdir_path,
    # количество эпох
    num_epochs=int(params["training"]["num_epochs"]),
    # вывод логирующих сообщений
    verbose=True,
)

exit(-1)