# pip install streamlit

import streamlit as st
import pandas as pd
import xgboost
import re
import numpy as np
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.metrics import f1_score
import pickle
from functools import partial
import pandas as pd
import numpy as np
import random
import re
import time
import datetime
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm, neighbors
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np

NELAGT_MODELS = {
    "Bagging_word_level": "For Unigram Tf-Idf feature vectors using Random Forest Classifier",
    "nela-gt-svm-uni": "",
    "article_transformers":"",
    "nela-gt-titles-svm":"",
    "Bagging_word_level_title":"",
    "rnn_lstm":"",
    "nela-gt-title-roberta": ""

}
COVID_MODELS = {"covid-svm-uni": "", "Bagging_word_level":"", "covid-roberta": ""}
LABELS = ['fake','real']
PYTORCH_ARTICLE_TRANSFORMERS_MODEL = None
DEVICE = torch.device("cpu")

def article_transformer(inp_text):
    
    tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-SST-2')
    model = PYTORCH_ARTICLE_TRANSFORMERS_MODEL
    
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = []
    def clean_text(text):   
        text = text.lower() # lowercase text
        text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
        text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
        text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
        return text
    
    def predict(article, model):
        article = clean_text(article)
        article.replace('\d+', '')
        input_ids = []
        attention_masks = []
        MAX_LENGTH = 220
        encoded_dict = tokenizer.encode_plus(
                            article,                      # Sentence to encode.
                            truncation=True,
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = MAX_LENGTH,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
    #     labels = torch.tensor(categories)
        batch_size = 1
        prediction_data = TensorDataset(input_ids, attention_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        model.eval()
        for batch in prediction_dataloader:
    # Add batch to GPU
            batch = tuple(t.to(DEVICE) for t in batch)
            b_input_ids, b_input_mask = batch
            with torch.no_grad():
            # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            return np.argmax(logits[0])

    def get_split(text1):
        l_total = []
        l_parcial = []
        if len(text1.split())//150 >0:
            n = len(text1.split())//150
        else: 
            n = 1
        for w in range(n):
            if w == 0:
                l_parcial = text1.split()[:200]
                l_total.append(" ".join(l_parcial))
            else:
                l_parcial = text1.split()[w*150:w*150 + 200]
                l_total.append(" ".join(l_parcial))
        return l_total

    def pipeline(article, model):
        splits = get_split(article)
        fake = 0
        real = 0
        for i in splits:
            if predict(i, model):
                fake += 1
            else:
                real += 1
        if real > fake:
            return "REAL"
        else:
            return "FAKE"

    return pipeline(inp_text, model)

def label_encode(val):
    return LABELS.index(val)

def clean_text(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = []
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text

def vectorize(t, input_text):
    out_vector = None
    pickle_path = "./models/" + t +"/tfidf_vect.pkl"

    with open(pickle_path, 'rb') as file:
        pickle_model = pickle.load(file)

    out_vector = pickle_model.transform(input_text)
    return out_vector

def vectorize_title(t, input_text):
    out_vector = None
    pickle_path = "./models/" + t +"/tfidf_vect_title.pkl"

    with open(pickle_path, 'rb') as file:
        pickle_model = pickle.load(file)

    out_vector = pickle_model.transform(input_text)
    return out_vector


dataset = st.sidebar.selectbox(
    'Choose a dataset',
     ["None"]+ ["covid","nelagt"], key="dataset")

models = None
if(dataset and dataset == "covid"):
    models = COVID_MODELS
elif(dataset and dataset == 'nelagt'):
    models = NELAGT_MODELS

if dataset != "None":
    option = st.sidebar.selectbox(
        'Choose a model',
        ["None"] + list(models.keys()), key = "model")




if(models):
    def load_return_model(t, model_name):
        pickle_path = "./models/"+ t +"/"+model_name+".pkl"
        with open(pickle_path, 'rb') as file:
            pickle_model = pickle.load(file)
        return pickle_model


    if option !=  "None":
        models[option]

        if st.checkbox('Predict Manual Text', key="manual_text"):
            inp_text = st.text_input("Input Text", value='', type='default')
            
            if(inp_text):
                if dataset=="nelagt" and option == "article_transformers":
                    with open('E:\\pramu\\projects\\final_project_fake_news_classsification\\Identification-of-fake-news-in-online-news-media\\streamlit\\models\\nelagt\\nela-gt-article-cascadingRoberta-epoch2.pt','rb') as f:
                        PYTORCH_ARTICLE_TRANSFORMERS_MODEL = torch.load(f, map_location = DEVICE)

                    predicted = article_transformer(inp_text)
                    predicted
                elif dataset=="nelagt" and option == "rnn_lstm":
                    new_model = tf.keras.models.load_model('E:\\pramu\\projects\\final_project_fake_news_classsification\\Identification-of-fake-news-in-online-news-media\\streamlit\\models\\nelagt\\rnn_lstm_v1.tf')
                    pred = new_model.predict_classes(np.array([inp_text]))
                    print(pred)
                    if pred == 0:
                        pred = "reliable"
                    else:
                        pred = "unreliabe"
                    pred
                elif dataset=="covid" and option == "covid-roberta":
                    with open('E:\\pramu\\projects\\final_project_fake_news_classsification\\Identification-of-fake-news-in-online-news-media\\streamlit\\models\\covid\\covid-roberta.pt','rb') as f:
                        PYTORCH_ARTICLE_TRANSFORMERS_MODEL = torch.load(f, map_location = DEVICE)

                    predicted = article_transformer(inp_text)
                    predicted
                elif dataset == "covid" and option == "nela-gt-title-roberta":
                    with open('E:\\pramu\\projects\\final_project_fake_news_classsification\\Identification-of-fake-news-in-online-news-media\\streamlit\\models\\nelagt\\nela-gt-title-roberta.pt','rb') as f:
                        PYTORCH_ARTICLE_TRANSFORMERS_MODEL = torch.load(f, map_location = DEVICE)

                    predicted = article_transformer(inp_text)
                    predicted
                else:
                    cleaned = [clean_text(inp_text)]
                    vectorized = vectorize(dataset, cleaned)
                    pickle_model = load_return_model(dataset, option)

                    Ypredict = pickle_model.predict(vectorized)
                    # proba = pickle_model.predict_proba(vectorized)
                    # proba
                    # explainer = LimeTextExplainer(class_names=[0,2])
                    print(Ypredict)
                    if dataset == "covid":
                        if Ypredict[0] == 0:
                            Ypredict = "fake"
                        else:
                            Ypredict = "real"
                    if dataset == "nelagt":
                        if Ypredict[0] == 0:
                            Ypredict = "reliable"
                        else:
                            Ypredict = "unreliable"
                    Ypredict
                    
                    # exp = explainer.explain_instance(cleaned, proba, num_features=6)
                    # exp = explainer.explain_instance(cleaned, pickle_model.predict_proba, num_features=6)


        if st.checkbox('Predict for n random samples'):
            n_sample = st.text_input("n:", value='', type='default', key='n_sample')
            if n_sample:  
                if dataset=="nelagt":
                    totalData = pd.read_csv('../nela10.csv')
                    if(option == "article_transformers"):
                        with open('E:\\pramu\\projects\\final_project_fake_news_classsification\\Identification-of-fake-news-in-online-news-media\\streamlit\\models\\nelagt\\nela-gt-article-cascadingRoberta-epoch2.pt','rb') as f:
                            PYTORCH_ARTICLE_TRANSFORMERS_MODEL = torch.load(f, map_location = DEVICE)
                        totalData = totalData.sample(n=int(n_sample))
                        totalData = totalData.drop(['id','date','source','title','author','url','published','published_utc','collection_utc'],axis=1)
                        totalData["Predicted"] = totalData.content.apply(article_transformer)
                        totalData
                    elif "title" in option: #TITLE
                        if option == "nela-gt-title-roberta":
                            with open('E:\\pramu\\projects\\final_project_fake_news_classsification\\Identification-of-fake-news-in-online-news-media\\streamlit\\models\\nelagt\\nela-gt-title-roberta.pt','rb') as f:
                                PYTORCH_ARTICLE_TRANSFORMERS_MODEL = torch.load(f, map_location = DEVICE)
                            totalData = totalData.sample(n=int(n_sample))
                            totalData = totalData.drop(['id','date','source','content','author','url','published','published_utc','collection_utc'],axis=1)
                            totalData["Predicted"] = totalData.title.apply(article_transformer)
                            totalData
                        else:    
                            totalData = totalData.sample(n=int(n_sample))
                            totalData = totalData.drop(['id','date','source','content','author','url','published','published_utc','collection_utc'],axis=1)
                            totalData.title = totalData.title.apply(clean_text)
                            totalData.title = totalData.title.str.replace('\d+', '')
                            pickle_model = load_return_model("nelagt", option)
                            totalData["Predicted_Reliability"] = pickle_model.predict(vectorize_title("nelagt", totalData["title"].tolist()))
                            totalData["Predicted_Reliability"] = totalData["Predicted_Reliability"].apply(lambda x : "reliable" if x == 0 else "unreliable")
                        totalData
                    elif option == "rnn_lstm":
                        totalData = totalData.sample(n=int(n_sample))
                        totalData = totalData.drop(['id','date','source','title','author','url','published','published_utc','collection_utc'],axis=1)
                        new_model = tf.keras.models.load_model('E:\\pramu\\projects\\final_project_fake_news_classsification\\Identification-of-fake-news-in-online-news-media\\streamlit\\models\\nelagt\\rnn_lstm_v1.tf')
                        totalData["predicted"] = new_model.predict_classes(totalData["content"].tolist())
                        totalData["predicted"] = totalData["predicted"].apply(lambda x : "reliable" if x == 0 else "unreliable")
                        totalData["Reliability"] = totalData["Reliability"].apply(lambda x : "reliable" if x == 0 else "unreliable")
                        totalData
                    else:
                        totalData = totalData.sample(n=int(n_sample))
                        totalData = totalData.drop(['id','date','source','title','author','url','published','published_utc','collection_utc'],axis=1)
                        totalData.content = totalData.content.apply(clean_text)
                        totalData.content = totalData.content.str.replace('\d+', '')
                        pickle_model = load_return_model("nelagt", option)
                        totalData["Predicted_Reliability"] = pickle_model.predict(vectorize("nelagt", totalData["content"].tolist()))
                        totalData["Reliability"] = totalData["Reliability"].apply(lambda x : "reliable" if x == 0 else "unreliable")

                        totalData["Predicted_Reliability"] = totalData["Predicted_Reliability"].apply(lambda x : "reliable" if x == 0 else "unreliable")
                        totalData
                elif dataset=="covid":
                    totalData = pd.read_csv('../Covid_Constraint_English_Train - Sheet1.csv')
                    totalData = totalData.sample(n=int(n_sample))
                    if option == "covid-roberta":
                        with open('E:\\pramu\\projects\\final_project_fake_news_classsification\\Identification-of-fake-news-in-online-news-media\\streamlit\\models\\covid\\covid-roberta.pt','rb') as f:
                            PYTORCH_ARTICLE_TRANSFORMERS_MODEL = torch.load(f, map_location = DEVICE)
                        totalData["Predicted"] = totalData.tweet.apply(article_transformer)
                        totalData["Predicted"] = totalData["Predicted"].apply(lambda x : "real" if x == "FAKE" else "fake")
                        totalData
                    else:
                        totalData.label = totalData.label.apply(label_encode)
                        totalData.tweet = totalData.tweet.apply(clean_text)
                        totalData.tweet = totalData.tweet.str.replace('\d+', '')
                        pickle_model = load_return_model("covid", option)
                        totalData["Predicted"] = pickle_model.predict(vectorize("covid", totalData["tweet"].tolist()))
                        totalData["label"] = totalData["label"].apply(lambda x : "real" if x == 1 else "fake")
                        totalData["Predicted"] = totalData["Predicted"].apply(lambda x : "real" if x == 1 else "fake")
                        totalData






