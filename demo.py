import gradio as gr
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

from DL2_BERT_Model_Based_Classification import CustomBertModel

SAVED_TARGET_CAT_PATH = "bbc-news-categories.torch"
# The actual is too large to be stored in github.
# So It is avaailable at the following URL : https://drive.google.com/file/d/1o-TDzHJwQfgw_y9R5PWo4TigkpYuSKmd/view?usp=sharing
SAVED_MODEL_PATH = "custom_bert_model.torch"
# SAVED_MODEL_PATH = "https://huggingface.co/jrmd/BERT-BASED-NEWS-CLASSIFICATION/blob/main/custom_bert_model.torch"


def find_category(
    input,
    saved_model_path=SAVED_MODEL_PATH,
    model_path="google-bert/bert-base-uncased",
    saved_target_cats_path=SAVED_TARGET_CAT_PATH,
):
    class_labels = torch.load(
        saved_target_cats_path, weights_only=False, map_location=torch.device("cpu")
    )
    saved_model = CustomBertModel(len(class_labels))
    saved_model.load_state_dict(
        torch.load(
            saved_model_path, weights_only=False, map_location=torch.device("cpu")
        )
    )  # Explicitly set weights_only to False

    saved_model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from path :{saved_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded_input = tokenizer.encode_plus(
        input,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    y_pred = ""
    with torch.no_grad():

        output = saved_model(
            encoded_input["input_ids"],
            encoded_input["attention_mask"],
        )

        y_pred = class_labels[output.squeeze(0).numpy().argmax()]

    return y_pred


with gr.Blocks() as demo:
    gr.Markdown(
        "<h1 style='text-align: center;'>CUSTOM MODEL BASED ON BERT BASE TO CLASSIFY NEWS ARTICLES</h1>"
    )
    gr.Markdown(
        "<h2 style='text-align: center;'>Model loss during training and eval time</h2>"
    )

    with gr.Row():
        gr.Image(value="wandb_chart_train.png", label="Training Loss")
        gr.Image(value="wandb_chart_eval.png", label="Eval Loss")

    gr.Interface(fn=find_category, inputs=["text"], outputs=["text"], live=False)

demo.launch(share=True)
