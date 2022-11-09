import matplotlib.pyplot as plt
import os
import logging
from pathlib import Path
from typing import List, Mapping, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml
from catalyst.utils import set_global_seed, prepare_cudnn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from catalyst.callbacks.metrics.accuracy import AccuracyCallback
import sklearn
from sklearn.metrics import accuracy_score
from catalyst.dl import (
    CheckpointCallback,
    OptimizerCallback,
    SchedulerCallback,
    SupervisedRunner,
)
import glob
import unicodedata
import seaborn as sns

pd.set_option('display.max_colwidth', 0)
sns.set(rc={'figure.figsize':(11,2)})

project_root: Path = Path("").parent.parent
base_path = os.path.abspath('')
data_path = os.path.join(base_path, 'DaNetQA')
def fileNameData(s):
    return f"{os.path.join(data_path, s)}.jsonl"

def loadJSONL(path, name = ""):
    b_print_info = len(name) > 0
    stat = None
    
    df = pd.read_json(path, lines=True)
    if b_print_info:
        print(name)
        display(df.head())
        pd_data = []
        for col in df.columns.values[0:2]:
            lenSymb = [len(x) for x in df[col]]
            pd_data.append([f"{col}(symbols)", np.min(lenSymb), np.max(lenSymb), int(np.mean(lenSymb)), lenSymb])
            lenTok = [len(word_tokenize(x)) for x in df[col]]
            pd_data.append([f"{col}(words)", np.min(lenTok), np.max(lenTok), int(np.mean(lenTok)), lenTok])
        stat = pd.DataFrame(pd_data, columns=['label', 'MIN', 'MAX', 'MEAN', 'LenArray'])
        print("Stats:")
        display(stat)


    # Y    
    if (df.columns.values == 'label').any():
        s = np.unique(df['label'].to_numpy(), return_counts=True)[1]
        if b_print_info:
            print(f"True answer: {s[1]}")
            print(f"False answer: {s[0]}")
            print("")

            
    return df, stat

def getConfigPaths(aL, aModels, aSeqSize, aParamsOptimize = ['PT']):
    configs_dict = dict()
    for iL in aL:
        for iM in aModels:
            for iS in aSeqSize:
                for iPO in aParamsOptimize:
                    name = f'L{iL}_M{iM}_S{iS}_{iPO}'
                    path = str(project_root / "configs" / f"config_{name}.yml")
                    if os.path.exists(path):
                        configs_dict[name] = path
    return configs_dict
configs_dict = getConfigPaths(
    range(0, 6), # уровни очистки данных с 0 (исходные) до 5 (все возможные уменьшения)
    range(1, 3), # проверяем на двух моделях
    [2**i for i in range(5, 10)], # размер кодировки: 32, 65, 128, 256, 512
    ['PT', 'PF']) # True и False значения

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
    context_column          = TEXT_FIELD_NAME
    question_column         = QUEST_FIELD_NAME
    label_column            = LABEL_FIELD_NAME
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

        # token_type_ids лежит во вложенном тензоре (x, 1', y), 
        # это мешает внутренним в Catalyst трансформациям поэтому вытягиваем его в (x, y)
        output_dict["token_type_ids"] = output_dict["token_type_ids"].squeeze(0)

        # encoding target
        if self.labels is not None:
            y = self.labels[index]
            y_encoded = torch.Tensor([self.label_dict.get(y, -1)]).long().squeeze(0)
            output_dict["targets"] = y_encoded
            
        return output_dict

class BertForSequenceClassification(nn.Module):
    """
    Simplified version of the same class by HuggingFace.
    See transformers/modeling_distilbert.py in the transformers repository.
    """

    def __init__(
        self, 
        pretrained_model_name: str, 
        num_classes: int = None, 
        dropout: float = 0.3,
    ):
        """
        Args:
            pretrained_model_name (str): HuggingFace model name.
                See transformers/modeling_auto.py
            num_classes (int): the number of class labels
                in the classification task
        """
        super().__init__()
        self.num_classes = num_classes
        config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=self.num_classes)

        self.model = AutoModel.from_pretrained(pretrained_model_name, config=config)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)

    def forward(self, features, attention_mask=None, token_type_ids=None, head_mask=None, targets=None):
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
        seq_output = bert_output[0]                     # (bs, seq_len, dim)
        # mean pooling, i.e. getting average representation of all tokens
        pooled_output = seq_output.mean(axis=1)         # (bs, dim)
        pooled_output = self.dropout(pooled_output)     # (bs, dim)
        scores = self.classifier(pooled_output)         # (bs, num_classes)
        
        #scores = self.softmax(scores)
        return scores

SEED = 17
NUM_CLASSES = 2
PATH_TO_LOG_FOLDER = Path('logdir')

TEXT_FIELD_NAME = 'passage'
QUEST_FIELD_NAME = 'question'
LABEL_FIELD_NAME = 'label'

df_validationL0, _ = loadJSONL(fileNameData("val_L0"))
y_val_true = df_validationL0.label.to_numpy().astype(np.int32)

df_testL0, _ = loadJSONL(fileNameData("test_L0"))

summary_path = "score_models.csv"
for config_key in configs_dict:
    for model_pth in glob.glob(f"logdir/{config_key}/model.*.pth"):
        predictPath = f"{model_pth}.predict.csv"
        if os.path.exists(predictPath) == False:

            with open(configs_dict[config_key]) as f:
                params = yaml.load(f, Loader=yaml.FullLoader)

            # загружаем датасет
            train_val_loaders, test_loaders = read_data(params)

            # загружаем модель из параметров с задангным кол-вом классов
            model = BertForSequenceClassification(
                pretrained_model_name=params["model"]["model_name"],
                num_classes=NUM_CLASSES
            )

            if params["training"]["optimize_parameters"] == True:
                param_optimizer = list(model.model.named_parameters())
                no_decay = ['bias', 'gamma', 'beta']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
                ]
            else:
                optimizer_grouped_parameters = model.parameters()

            # and running inference
            torch.cuda.empty_cache()

            runner = SupervisedRunner(input_key=("features", "attention_mask", "token_type_ids"))

            Y_predictFt = []
            for prediction in runner.predict_loader(
                model=model,
                loader=train_val_loaders['valid'],
                resume = model_pth
            ):
                Y_predictFt.extend(prediction["logits"].detach().cpu().numpy())
            
            y_predict = np.argmax(Y_predictFt, axis=1)
            np.savetxt(predictPath, y_predict, delimiter=';')

            score = accuracy_score(y_val_true, y_predict)
            with open(summary_path, 'a') as file:
                file.write(f"{config_key};{model_pth};{score}\n")
            print(f"Config={config_key} model={model_pth} score={score}")
            exit(20)
exit(1)