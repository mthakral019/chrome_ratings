from distutils.command.upload import upload
from flask import Flask,render_template,request,redirect,url_for
import os
from os.path import join,dirname,realpath
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


app=Flask(__name__)


UPLOAD_FOLDER='static/files'
app.config['UPLOAD_FOLDER']= UPLOAD_FOLDER



@app.route('/')
def index():
    return render_template("index.html")

@app.route("/",methods=["POST"])
def uploadFiles():
    uploaded_file=request.files["file"]
    if uploaded_file.filename!="":
        file_path=os.path.join(app.config['UPLOAD_FOLDER'],uploaded_file.filename)
        uploaded_file.save(file_path)

        def parseCV(file_path):
            data=pd.read_csv(file_path)
            df= data[['Text','Star']]
            df['Star'].astype('int')
            df['is_rating_bad']=df['Star'].apply(lambda x: 1 if x<3 else 0)

            df.dropna(inplace=True)

            sid=SentimentIntensityAnalyzer()
            df['score']=df['Text'].apply(lambda review: sid.polarity_scores(review))

            df['positive']=df['score'].apply(lambda score_dict:score_dict['pos'])
            df['negative']=df['score'].apply(lambda score_dict:score_dict['neg'])
            df['neutral']=df['score'].apply(lambda score_dict:score_dict['neu'])
            df['compound']=df['score'].apply(lambda score_dict:score_dict['compound'])
            df['bad_pred']=df['compound'].apply(lambda x: 0 if x>=0 else 1)

            index_list=df[(df['bad_pred']==0) & (df['is_rating_bad']==1)&(df['neutral']<0.5)].index
            new_data=pd.DataFrame([],columns=data.columns)
            enteries=[]
            for i in index_list:
                entry=data.iloc[data.index==i]
                enteries.append(entry)
            
            new_data=pd.concat(enteries)
            new_data.reset_index(drop=True)

            return new_data

        new_data= parseCV(file_path)
    return render_template('result.html',data=new_data)


if __name__=="__main__":
    app.run(debug=False,port=5000)