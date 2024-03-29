from ast import keyword
import re
import nltk
import emoji
from nltk import inference
import tweepy
from flask import Flask, request, render_template
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
import re
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import TFRobertaModel
from transformers import RobertaTokenizerFast
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from flask import jsonify, request, render_template
from geopy.geocoders import Nominatim




def preprocess(sentence):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    sentence = re.sub('[^A-z]', ' ', sentence)
    negative = ['not', 'neither', 'nor', 'but', 'however', 'although', 'nonetheless', 'despite', 'except',
                        'even though', 'yet']
    stop_words = [z for z in stop_words if z not in negative]
    preprocessed_tokens = [lemmatizer.lemmatize(contractions.fix(temp.lower())) for temp in sentence.split() if temp not in stop_words] #lemmatization
    return ' '.join([x for x in preprocessed_tokens]).strip()

train_data = pd.read_csv('train.txt', names=['text', 'emotion'], sep=';')
val_data = pd.read_csv('val.txt', names=['text', 'emotion'], sep=';')
test_data = pd.read_csv('test.txt', names=['text', 'emotion'], sep=';')

train_data['text'] = train_data['text'].apply(lambda x: preprocess(x))
val_data['text'] = val_data['text'].apply(lambda x: preprocess(x))
test_data['text'] = test_data['text'].apply(lambda x: preprocess(x))

data = {'Train Data': train_data, 'Validation Data': val_data, 'Test Data': test_data}   


ros = RandomOverSampler(random_state=0)
train_x, train_y = ros.fit_resample(np.array(train_data['text']).reshape(-1, 1), np.array(train_data['emotion']).reshape(-1, 1))
train = pd.DataFrame(list(zip([x[0] for x in train_x], train_y)), columns = ['text', 'emotion'])



le = preprocessing.OneHotEncoder()
y_train= le.fit_transform(np.array(train['emotion']).reshape(-1, 1)).toarray()
y_test= le.fit_transform(np.array(test_data['emotion']).reshape(-1, 1)).toarray()
y_val= le.fit_transform(np.array(val_data['emotion']).reshape(-1, 1)).toarray()



tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

def roberta_encode(data,maximum_length) :
  input_ids = []
  attention_masks = []
  

  for i in range(len(data.text)):
      encoded = tokenizer.encode_plus(
        
        data.text[i],
        add_special_tokens=True,
        max_length=maximum_length,
        pad_to_max_length=True,
        
        return_attention_mask=True,
        
      )
      
      input_ids.append(encoded['input_ids'])
      attention_masks.append(encoded['attention_mask'])
  return np.array(input_ids),np.array(attention_masks)

max_len = max([len(x.split()) for x in train_data['text']])
train_input_ids,train_attention_masks = roberta_encode(train, max_len)
test_input_ids,test_attention_masks = roberta_encode(test_data, max_len)
val_input_ids,val_attention_masks = roberta_encode(val_data, max_len)

def create_model(bert_model, max_len):
    input_ids = tf.keras.Input(shape=(max_len,),dtype='int32')
    attention_masks = tf.keras.Input(shape=(max_len,),dtype='int32')

    output = bert_model([input_ids,attention_masks])
    output = output[1]

    output = tf.keras.layers.Dense(6, activation='softmax')(output)
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


roberta_model = TFRobertaModel.from_pretrained('roberta-base')

model = create_model(roberta_model, max_len)


def plot_result(result):
    sns.barplot(x = 'Category', y = 'Confidence', data = result)
    plt.xlabel('Categories', size=14)
    plt.ylabel('Confidence', size=14)
    plt.title('Emotion Classification', size=16)

def roberta_inference_encode(data,maximum_length) :
    input_ids = []
    attention_masks = []
  

  
    encoded = tokenizer.encode_plus(
    data,
    add_special_tokens=True,
    max_length=maximum_length,
    pad_to_max_length=True,

    return_attention_mask=True

    )

    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)

def inference(text_sentence, max_len):
    preprocessed_text = preprocess(text_sentence)
    input_ids, attention_masks = roberta_inference_encode(preprocessed_text, maximum_length = max_len)
    model = create_model(roberta_model, 43)
    model.load_weights('my_checkpoint')
    result = model.predict([input_ids, attention_masks])
    #le.categories_[0] = ['anger' 'fear' 'joy' 'love' 'sadness' 'surprise']
    result = pd.DataFrame(dict(zip(list(le.categories_[0]), [round(x*100, 2)for x in result[0]])).items(), columns = ['Category', 'Confidence'])
    column = result["Confidence"]
    max_index = column.idxmax()
    emotion = result["Category"].iloc[max_index]
    return emotion


nltk.download('vader_lexicon')
geolocator = Nominatim(user_agent="SEA")
vader_analyzer = SentimentIntensityAnalyzer()
app = Flask(__name__)

consumer_key = "3ahhpzS1BrmNOBFhrPmKA1Vsd"
consumer_secret = "CFACv1AHjZEs5vP8PJ0lPtWUiqTtI5hyA0rXK1Zq1DXZdJrrca"
access_token = "3191865854-ZoqzzmOdaWQQG1TX67x18utsnHM1n7ZkKrIkOHX"
access_token_secret = "7EAF5mTzI2FzSAr2jS8g6NUTuFLmJSAtZq4yuObjjOALH"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
client = tweepy.Client()


def cleaning(text):
    text = re.sub("@[A-Za-z0-9_]+", "", text)
    text = re.sub("#[A-Za-z0-9_]+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www.\S+", "", text)
    return text


def sentimental_analyze(text):
    result = ""
    sentiment = ""
    if emoji.emoji_count(text) != 0:
        result = vader_analyzer.polarity_scores(emoji.demojize(text))
    else:
        result = vader_analyzer.polarity_scores(text)

    if result["neu"] == max(result["neu"], result["pos"], result["neg"]):
        sentiment = "Neutral"
    elif result["pos"] == max(result["neu"], result["pos"], result["neg"]):
        sentiment = "Positive"
    else:
        sentiment = "Negative"
    return  text 
    
def sentimental_analyz(text):
    result = ""
    sentiment = ""
    if emoji.emoji_count(text) != 0:
        result = vader_analyzer.polarity_scores(emoji.demojize(text))
    else:
        result = vader_analyzer.polarity_scores(text)
    if result["pos"]<=0.15 and result["neg"]<=0.15:
        sentiment = "Neutral"
    else:
        if result["pos"] == max(result["pos"], result["neg"]):
            sentiment = "Positive"
        if result["neg"] == max(result["pos"], result["neg"]):
            sentiment = "Negative"
    return sentiment

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/sen_home")
def sen_home():
    return render_template("sen_home.html")

@app.route("/input_snt")
def bbs():
    return render_template("input_snt.html")


@app.route("/result", methods=["POST"])
def result():

    article = request.form["article"]
    clean_text = cleaning(article)
    lines = sentimental_analyze(clean_text)
    sentiment= sentimental_analyz(clean_text)
    return render_template("input_snt.html",sentiment=sentiment, lines=lines)


@app.route("/twitter_home")
def twitter_home():
    return render_template("twitter_home.html")


@app.route("/twitter_keyword")
def twitter_keyword():
    return render_template("twitter_keyword.html")


@app.route("/twitter_id")
def twitter_id():
    return render_template("twitter_id.html")


@app.route("/twitter_url")
def twitter_url():
    return render_template("twitter_url.html")
@app.route("/emo_home")
def emo_home():
    return render_template("emo_home.html")
@app.route("/input_emo")
def input_emo():
    return render_template("input_emo.html")
@app.route("/result_emo", methods=["POST"])
def result_emo():
    article = request.form["article"]
    clean_text = cleaning(article)
    lines = sentimental_analyze(clean_text)
    sentiment= inference(clean_text,max_len)
    return render_template("input_emo.html",sentiment=sentiment, lines=lines)
@app.route("/twitter_home_emo")
def twitter_home_emo():
    return render_template("twitter_home_emo.html")
@app.route("/twitter_keyword_emo")
def twitter_keyword_emo():
    keyword='piechart'
    data = {'Task' : 'Hours per Day','anger' : 1, 'fear' : 1, 'joy' : 1, 'love' : 1, 'sadness' : 1, 'surprise' : 1}
    return render_template("twitter_keyword_emo.html", data=data, Title=keyword)
@app.route("/twitter_id_emo")
def twitter_id_emo():
    T_id='piechart'
    data = {'Task' : 'Hours per Day','anger' : 1, 'fear' : 1, 'joy' : 1, 'love' : 1, 'sadness' : 1, 'surprise' : 1}
    return render_template("twitter_id_emo.html", data=data ,Title=T_id)
@app.route("/twitter_url_emo")
def twitter_url_emo():
    Title='piechart'
    data = {'Task' : 'Hours per Day','anger' : 1, 'fear' : 1, 'joy' : 1, 'love' : 1, 'sadness' : 1, 'surprise' : 1}
    return render_template("twitter_url_emo.html", data=data,Title=Title)

@app.route("/twitter_keyword_result", methods=["POST"])
def twitter_keyword_result():
    tweets = []
    keyword = request.form["keyword"]
    location = request.form["location"]
    T_num = request.form["num"]
    keyword = keyword + " exclude:retweets -filter:links"
    geocode_location = geolocator.geocode(location)
    if(len(keyword)==0 or len(T_num)==0):
        error = "Please supply both first and last name"
    else:
        if(len(location) == 0):
            for tweet in tweepy.Cursor(api.search_tweets, q=keyword).items(int(T_num)):
                clean_text = cleaning(tweet.text)
                lines=sentimental_analyze(clean_text)
                tweets.append((lines,sentimental_analyz(clean_text)))
        else:
            for tweet in tweepy.Cursor(api.search_tweets, q=keyword, geocode=str(geocode_location.latitude) + "," + str(geocode_location.longitude) + ",100km").items(int(T_num)):
                clean_text = cleaning(tweet.text)
                lines=sentimental_analyze(clean_text)
                tweets.append((lines,sentimental_analyz(clean_text)))
    return render_template("twitter_keyword.html", tweets = tweets)

@app.route("/twitter_id_result", methods=["POST"])
def twitter_id_result():
    tweets = []
    T_id = request.form["id"]
    T_num = request.form["num"]
    if(len(T_id)==0 or len(T_num)==0):
        error = "Please supply both first and last name"
    else:
        for tweet in tweepy.Cursor(api.user_timeline, screen_name=T_id, tweet_mode='extended').items(int(T_num)):
            clean_text = cleaning(tweet.full_text)
            lines = sentimental_analyze(clean_text)
            tweets.append((lines,sentimental_analyz(clean_text)))
    return render_template("twitter_id.html", tweets=tweets)
def counter(emotions):
    joy_count=emotions.count("joy")
    anger_count=emotions.count("anger")
    love_count=emotions.count("love")
    surprise_count=emotions.count("surprise")
    fear_count=emotions.count("fear")
    sadness_count=emotions.count("sadness")
    data = {'Task' : 'Hours per Day','anger':anger_count, 'fear':fear_count, 'joy':joy_count, 'love':love_count, 'sadness':sadness_count, 'surprise':surprise_count}
    return data

@app.route("/twitter_keyword_emo_result", methods=["POST"])
def twitter_keyword_emo_result():
    if request.method == 'POST':
        tweets = []
        emotions = []
        keyword = request.form["keyword"]
        Title=keyword
        location = request.form["location"]
        T_num = request.form["num"]
        keyword = keyword + " exclude:retweets -filter:links"
        geocode_location = geolocator.geocode(location)
        if len(keyword) == 0 or len(T_num)== 0:
             # Form data failed validation; try again
             error = "Please supply both first and last name"
        else:
            if(len(location) == 0):
                for tweet in tweepy.Cursor(api.search_tweets, q=keyword).items(int(T_num)):
                    clean_text = cleaning(tweet.text)
                    lines = sentimental_analyze(clean_text)
                    emotion=inference(clean_text,max_len)
                    emotions.append(emotion)
                    tweets.append((lines,emotion))
                data=counter(emotions)
            else:
                for tweet in tweepy.Cursor(api.search_tweets, q=keyword, geocode=str(geocode_location.latitude) + "," + str(geocode_location.longitude) + ",100km").items(int(T_num)):
                    clean_text = cleaning(tweet.text)
                    lines = sentimental_analyze(clean_text)
                    emotion=inference(clean_text,max_len)
                    emotions.append(emotion)
                    tweets.append((lines,emotion))
                data=counter(emotions)
        return render_template("twitter_keyword_emo.html", tweets=tweets, data=data, Title=Title)
    
@app.route("/twitter_url_result", methods=["POST"])
def twitter_url_result():
    tweet = []
    T_url = request.form["url"]
    if(len(T_url)==0):
        error = "Please supply both first and last name"
    else:
        tweet_id = T_url.split('status/')[-1].split('?')[0]
        clean_text = cleaning(api.get_status(tweet_id).fulltext)
        lines = sentimental_analyze(clean_text)
        tweet.append((lines,sentimental_analyz(clean_text)))
    return render_template("twitter_url.html", tweet=tweet)

@app.route("/twitter_url_emo_result", methods=["POST"])
def twitter_url_emo_result():
    if request.method == 'POST':
         tweets = []
         emotions=[]
         T_url = request.form["url"]
         if len(T_url) == 0:
             # Form data failed validation; try again
             error = "Please supply both first and last name"
         else:
             tweet_id = T_url.split('status/')[-1].split('?')[0]
             clean_text = cleaning(api.get_status(tweet_id).fulltext)
             lines = sentimental_analyze(clean_text)
             emotion=inference(clean_text,max_len)
             emotions.append(emotion)
             tweets.append((lines,emotion))
             joy_count=emotions.count("joy")
             anger_count=emotions.count("anger")
             love_count=emotions.count("love")
             surprise_count=emotions.count("surprise")
             fear_count=emotions.count("fear")
             sadness_count=emotions.count("sadness")
             data = {'Task' : 'Hours per Day','anger':anger_count, 'fear':fear_count, 'joy':joy_count, 'love':love_count, 'sadness':sadness_count, 'surprise':surprise_count}
         return render_template("twitter_url_emo.html", tweets=tweets, data=data,Title=T_url)

@app.route("/twitter_id_emo_result", methods=["POST"])
def twitter_id_emo_result():
    if request.method == 'POST':
        tweets = []
        emotions=[]
        T_id = request.form["id"]
        T_num = request.form["num"]
        if len(T_id) == 0 or len(T_num) == 0:
            # Form data failed validation; try again
             error = "Please supply both first and last name"
        else:
            for tweet in tweepy.Cursor(api.user_timeline, screen_name=T_id, tweet_mode='extended').items(int(T_num)):
                clean_text = cleaning(tweet.full_text)
                lines = sentimental_analyze(clean_text)
                emotion=inference(clean_text,max_len)
                emotions.append(emotion)
                tweets.append((lines,emotion))
            data=counter(emotions)
        return render_template("twitter_id_emo.html", tweets=tweets, data=data, Title=T_id)


		
