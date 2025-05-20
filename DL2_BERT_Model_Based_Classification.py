import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, BertModel

import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 5
BATCH_SIZE = 16
SAVED_MODEL_PATH = "custom_bert_model.torch"
SAVED_TARGET_CAT_PATH = "bbc-news-categories.torch"
DS_PATH = "bbc-news-data.csv"


from typing import DefaultDict


class CustomBertDataset(Dataset):
    def __init__(
        self,
        file_path,
        model_path="google-bert/bert-base-uncased",
        saved_target_cats_path=SAVED_TARGET_CAT_PATH,
    ):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.lines = open(file_path).readlines()
        self.lines = np.array(
            [
                [
                    re.split(r"\t+", line.replace("\n", ""))[3],
                    re.split(r"\t+", line.replace("\n", ""))[0],
                ]
                for i, line in enumerate(self.lines)
                if line != "\n" and i != 0
            ]
        )
        self.corpus = np.array(self.lines[:, 0])
        self.elem_cats = self.lines[:, 1]
        self.unique_cats = sorted(list(set(self.elem_cats)))
        self.num_class = len(self.unique_cats)
        self.cats_dict = {cat: i for i, cat in enumerate(self.unique_cats)}
        self.targets = np.array([self.cats_dict[cat] for cat in self.elem_cats])

        torch.save(self.unique_cats, saved_target_cats_path)

        entry_dict = DefaultDict(list)
        for i in range(len(self.corpus)):
            entry_dict[self.targets[i]].append(self.corpus[i])

        self.final_corpus = []
        self.final_targets = []
        n = 0
        while n < len(self.corpus):
            for key in entry_dict.keys():
                if len(entry_dict[key]) > 0:
                    self.final_corpus.append(entry_dict[key].pop(0))
                    self.final_targets.append(key)
                    n += 1

        self.corpus = np.array(self.final_corpus)
        self.targets = np.array(self.final_targets)

        self.max_len = 0
        for sent in self.corpus:
            input_ids = self.tokenizer.encode(sent, add_special_tokens=True)
            self.max_len = max(self.max_len, len(input_ids))

        self.max_len = min(self.max_len, 512)
        print(f"Max length : {self.max_len}")

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        text = self.corpus[idx]
        target = self.targets[idx]
        encoded_input = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            encoded_input["input_ids"].squeeze(0),
            encoded_input["attention_mask"].squeeze(0),
            torch.tensor(target, dtype=torch.long),
        )
        # return np.array(encoded_input), torch.tensor(target, dtype=torch.long)


class CustomBertModel(nn.Module):
    def __init__(self, num_class, model_path="google-bert/bert-base-uncased"):
        super(CustomBertModel, self).__init__()
        self.model_path = model_path
        self.num_class = num_class

        self.bert = BertModel.from_pretrained(self.model_path)
        # Freeze of the parameters of this layer for the training process
        for param in self.bert.parameters():
            param.requires_grad = False
        self.proj_lin = nn.Linear(self.bert.config.hidden_size, self.num_class)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        x = x.last_hidden_state[:, 0, :]
        x = self.proj_lin(x)
        return x


def train_step(model, train_dataloader, loss_fn, optimizer):

    num_iterations = len(train_dataloader)

    for i in range(NUM_EPOCHS):
        print(f"Training Epoch n° {i}")
        model.train()

        for j, batch in enumerate(train_dataloader):

            input = batch[:][0]
            attention = batch[:][1]
            target = batch[:][2]

            output = model(input.to(device), attention.to(device))

            loss = loss_fn(output, target.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run.log({"Training loss": loss})

            print(f"Epoch {i+1} | step {j+1} / {num_iterations} | loss : {loss}")

    # Save model
    torch.save(model.state_dict(), SAVED_MODEL_PATH)
    print(f"Model saved at {SAVED_MODEL_PATH}")


def eval_step(
    test_dataloader,
    loss_fn,
    num_class,
    saved_model_path=SAVED_MODEL_PATH,
    saved_target_cats_path=SAVED_TARGET_CAT_PATH,
):

    y_pred = []
    y_true = []

    num_iterations = len(test_dataloader)
    # Load the saved model
    saved_model = CustomBertModel(num_class)
    saved_model.load_state_dict(
        torch.load(saved_model_path, weights_only=False)
    )  # Explicitly set weights_only to False
    saved_model = saved_model.to(device)
    saved_model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from path :{saved_model_path}")

    with torch.no_grad():
        for j, batch in enumerate(test_dataloader):

            input = batch[:][0]
            attention = batch[:][1]
            target = batch[:][2]

            output = saved_model(input.to(device), attention.to(device))

            loss = loss_fn(output, target.to(device))

            run.log({"Eval loss": loss})
            print(f"Eval loss : {loss}")
            y_pred.extend(output.cpu().numpy().argmax(axis=1))
            y_true.extend(target.cpu().numpy())

    class_labels = torch.load(saved_target_cats_path, weights_only=False)

    true_labels = [class_labels[i] for i in y_true]
    pred_labels = [class_labels[i] for i in y_pred]

    print(f"Accuracy : {accuracy_score(true_labels, pred_labels)}")

    cm = confusion_matrix(true_labels, pred_labels, labels=class_labels)
    df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    sns.heatmap(df_cm, annot=True, fmt="d")
    plt.title("Confusion Matrix for BBC News Dataset")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


if __name__ == "__main__":

    wandb.login()
    run = wandb.init(project="DIT-Bert-bbc-news-project")
    our_bert_dataset = CustomBertDataset(DS_PATH)
    print(f"Size of bert dataset : {len(our_bert_dataset)}")
    train_dataset = Subset(our_bert_dataset, range(int(len(our_bert_dataset) * 0.8)))
    test_dataset = Subset(
        our_bert_dataset, range(int(len(our_bert_dataset) * 0.8), len(our_bert_dataset))
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    our_bert_model = CustomBertModel(our_bert_dataset.num_class)
    our_bert_model = our_bert_model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, our_bert_model.parameters()), lr=0.01
    )

    train_step(our_bert_model, train_dataloader, loss_fn, optimizer)

    eval_step(test_dataloader, loss_fn, our_bert_dataset.num_class)
