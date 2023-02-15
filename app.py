import pandas as pd
import numpy as np
from flask import Flask,request,render_template
from flask_cors import cross_origin
import pickle

app = Flask(__name__)


model=pickle.load(open('model3.pkl','rb'))

@app.route('/')
@cross_origin()
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST','GET'])
@cross_origin()
def predict():
    if request.method == 'POST':
        age=int(request.form['Age'])
        sex=int(request.form['Sex'])
        cp=int(request.form['Chest pain']) 
        chol=int(request.form['Cholestrol'])
        fbs=int(request.form['fbs'])
        thalach=int(request.form['hattack'])
        exang=int(request.form['exang'])
        oldpeak=int(request.form['oldpeak'])
        ca=int(request.form['ca'])
        array=np.array([[age,sex,cp,chol,fbs,thalach,exang,oldpeak,ca]])

        features=pd.DataFrame(array, columns = ['age','sex','cp','chol','fbs','thalach','exang','oldpeak','ca'])
        pred=model.predict(features)
        if pred[0]:
            op="Chance of getting Heart Disease"
        else:
            op="Cheers!!.. No Diseases"
        # op="Cheers!!.. No Diseases"
        return render_template('home.html',output_text = op,Age=age,sex=sex,chest_pain=cp,Cholestrol=chol,fbs=fbs,hattack=thalach,exang=exang,oldpeak=oldpeak,ca=ca)
        #   return render_template('home.html')
    return render_template('home.html')
    
if __name__ == "__main__":
    app.run(debug=True) 
    
    
    
    
