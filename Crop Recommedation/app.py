from flask import Flask,render_template,request,redirect,url_for
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask(__name__)
model = joblib.load("_model_.joblib")
label = {0: 'rice', 1: 'maize', 2: 'chickpea', 3: 'kidneybeans', 4: 'pigeonpeas', 5: 'mothbeans', 6: 'mungbean', 7: 'blackgram', 8: 'lentil', 9: 'pomegranate', 10: 'banana', 11: 'mango', 12: 'grapes', 13: 'watermelon', 14: 'muskmelon', 15: 'apple', 16: 'orange', 17: 'papaya', 18: 'coconut', 19: 'cotton', 20: 'jute', 21: 'coffee'}
df = pd.read_csv("Crop_recommendation.csv")

@app.route('/')
def index():
    prediction_value = request.args.get("prediction_value")
    return render_template("index.html",prediction_value=prediction_value)

@app.route('/predict',methods=["POST"])
def predict():
    N:int = int(request.form["N"])
    P:int = int(request.form["P"])
    K:int = int(request.form["K"])
    temperature:float= float(request.form["temperature"])
    humidity:float = float(request.form["humidity"])
    pH:float = float(request.form["pH"])
    rainfall:float = float(request.form["rainfall"])
    feature = [N,P,K,temperature,humidity,pH,rainfall]
    scaler = MinMaxScaler()
    scaler.fit(df.drop("label",axis=1))
    feature = scaler.transform([feature])
    prediction_value:int = model.predict(feature)[0]
    answer:str = label[prediction_value]
    return redirect(url_for("index",prediction_value=answer))

if __name__ == "__main__":
    app.run(debug=True)