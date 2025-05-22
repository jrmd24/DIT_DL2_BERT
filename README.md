---
license: apache-2.0
title: NEWS ARTICLE CLASSIFIER BASED ON BERT MODEL
sdk: gradio
app_file: demo.py
pinned: false
---

# NEWS ARTICLE CLASSIFIER BASED ON BERT MODEL

This contains the code to train a pretrained BERT Base model to classify news articles into categories like "sport", "politics", or "tech". The BBC news dataset was used to perform the training. 
The weights within the actual BERT base model were frozen during the training process. We were just fine-tuning the linear fully connected layer used to determine the final class of articles.

The Training is performed by the "DL2_BERT_Model_Based_Classification.py" python script. A notebook with the same name but with the ".ipynb" extension is available at the same location. The resulting model has been upload in the following Hugging Face model repository : jrmd/BERT-BASED-NEWS-CLASSIFICATION

The "Demo.py" is a gradio application showing the graph of the losses collected during training and evaluation of the model. It also allows, right under the graphs, to input an news article and get its predicted category after submission. We can see the result in the screenshot below.

![Alt text](HomeGradio.png)

To run this on your computer :
- Clone the repo : git clone https://github.com/jrmd24/DIT_DL2_BERT.git
- Download the model (https://huggingface.co/jrmd/BERT-BASED-NEWS-CLASSIFICATION/resolve/main/custom_bert_model.torch?download=true) and place it in the same folder as the "demo.py" file
- Install gradio : pip install gradio # It is recommended to install it in a python virtual environment
- Run the app : python demo.py
- Navigate on your computer at the following url : http://localhost:7860
- Input news article text in the "input" field then click on the "submit"

An online link allows to access the app, but the inference might not be functional on it yet : https://jrmd-bert-based-news-classification.hf.space/?logs=container&__theme=system&deep_link=bpiyD9W-I0k

