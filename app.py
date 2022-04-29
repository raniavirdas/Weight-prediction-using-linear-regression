from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def height_prediction():
    if request.method == 'GET':
        return render_template("weight-height.html")
    elif request.method == 'POST':
        model = joblib.load("model-development/model_linearregression.pkl")
        Gender_Male = {
            'Female': 0,
            'Male': 1
        }
        features = dict(request.form)
        features = np.array([Gender_Male[features['Gender_Male']],float(features['Height'])])
        result = model.predict([features])
        result = str(round(result[0],2))
        return render_template('weight-height.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=500, debug=True)