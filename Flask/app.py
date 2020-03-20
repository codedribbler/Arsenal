from flask import Flask, render_template,request
import numpy as np
import pickle
app = Flask(__name__)


model = pickle.load(open('regmodel.pkl','rb'))
    
@app.route('/')
def home():
    return render_template("viv.html")
@app.route('/result',methods=['POST','GET'])
def result():
    if request.method == 'GET':
        return 'Data Entered'
    elif request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        return str(output)
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')