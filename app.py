
import numpy as np
from flask import Flask,request,jsonify,render_template,url_for
import pickle
app = Flask(__name__)

sc=pickle.load(open('sc.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    l=[]
    age=int(request.form('Age'))
    l.append(age)
    sex=int(request.form('Sex'))
    l.append(sex)
    cp=int(request.form('Chest pain Type')) 
    l.append(cp)
    chol=int(request.form('Cholestrol'))
    l.append(chol)
    fbs=int(request.form('Fasting Blood Sugar'))
    l.append(fbs)
    thalach=int(request.form('Max heartrate'))
    l.append(thalach)
    exang=int(request.form('Exang'))
    l.append(exang)
    oldpeak=int(request.form('oldpeak'))
    l.append(oldpeak)
    features=np.array(l)
    pred=model.predict(sc.transform(features))
    return render_template('result.html',prediction=pred)
    
if __name__ == "__main__":
    app.run(debug=True) 
    
    
    
    