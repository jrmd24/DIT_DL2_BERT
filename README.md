---
title: News_article_classifier_based_on_BERT_model
app_file: demo.py
sdk: gradio
sdk_version: 5.29.1
---

This contains the code to train a pretrained BERT Base model to classify news articles into categories like "sport", "politics", or "tech". The BBC news dataset was used to perform the training. 
The weights within the actual BERT base model were frozen during the training process. We were just fine-tuning the linear fully connected layer used to determine the final class of articles.

The Training is performed by the "DL2_BERT_Model_Based_Classification.py" python script. A notebook with the same name but with the ".ipynb" extension is available at the same location. The resulting model has been upload in the following Hugging Face model repository : jrmd/BERT-BASED-NEWS-CLASSIFICATION

The "Demo.py" is a gradio application showing the graph of the losses collected during training and evaluation of the model. It also allows, right under the graphs, to input an news article and get its predicted category after submission. We can see the result in the screenshot below.

![Alt text](HomeGradio.png)

L'application de démonstration du modèle est accessible via le lien : https://jrmd-bert-based-news-classification.hf.space/?logs=container&__theme=system&deep_link=bpiyD9W-I0k