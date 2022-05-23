
from flask import Flask, request, render_template
import pickle
from markupsafe import Markup

# app = Flask(__name__): Creates an instance of the Flask class. __name__ is a variable that represents the name of the application’s module (this helps Flask know where to look for resources like “templates” which we’ll use later)
app = Flask(__name__)

# Load the models and vectorizer that you saved with pickle
SVC_model = pickle.load(open('models-and-flask-app/SVC.pkl', 'rb'))
LR_model = pickle.load(open('models-and-flask-app/LOGISTICREGRESSION.pkl','rb'))
RandomForest_model = pickle.load(open('models-and-flask-app/RANDOMFORESTCLASSIFIER.pkl','rb'))

# Create a TF-IDF representation of the text
def data_iterator(f):
    for token in f:
        yield token


def tokenizer(txt):
    """Simple whitespace tokenizer"""
    return txt.split()

vectorizer=pickle.load(open("models-and-flask-app/vectorizer.pickle", 'rb'))     #Load vectorizer

#@app.route("/"): @ represents decorators (they modify the behavior of a function or class). The route() decorator tells Flask what URL should trigger our function. In our example, the homepage (/) should trigger the hello_world() function.
@app.route("/")

# This function will render an HTML file named index.html, every time we visit the homepage (‘/’). You need to create a HTML file for the front-end of this page. To build the front end, first, create a folder named “templates” (you can’t use any other name) in the folder where your app is located. Inside this “templates” folder create an HTML file. Open the HTML file and write in "html" and press the tab. This wil create the basic HTML structure. Now to add bootstrap, go to this website (https://getbootstrap.com/docs/5.1/getting-started/introduction/) and copy/paste the code inside the CSS section into the <head> section of index.html. This will load their CSS.

# We’ll also add a navigation bar, so go to the navbar section of the website (https://getbootstrap.com/docs/5.1/components/navbar/)and copy the code and paste it into the <head> section (right below the CSS)

def home():
    return render_template('index.html')

#Now let’s build the predict function! Inside this (.py) file, we create a decorator with URL “/predict” that triggers a predict function and renders the index.html file.

# We need to import request from flask to get access to the values the end-user introduced in the form.

# The values inside the form are returned with a dictionary shape, so we have to use square brackets to obtain each value. Once we have the values we put them inside double square brackets [[ ]] to make the predictions (this is the format our model.pkl accepts)

@app.route('/predict',methods=['POST'])

def predict():
    """Grabs the input values and uses them to make prediction"""
    predict_text = str(request.form["input_text"])

    test_iterator=data_iterator([predict_text])

    d_test=vectorizer.transform(test_iterator)

    SVC_prediction = SVC_model.predict(d_test)  # this returns a list e.g. [127.20488798], so pick first element [0]
    SVC_output = SVC_prediction[0] 
    # Predict probabilities
    SVC_probabilities = SVC_model.predict_proba(d_test)

    LR_prediction = LR_model.predict(d_test)
    LR_output = LR_prediction[0]
    # Predict probabilities
    LR_probabilities = LR_model.predict_proba(d_test)

    RF_prediction = RandomForest_model.predict(d_test)
    RF_output = RF_prediction[0]
    RF_probabilities = RandomForest_model.predict_proba(d_test)

    return render_template('index.html', prediction_text=Markup(f'The Support Vector Machine model classified this text as: {SVC_output}.<br/>Probabilities for "objective": {round(SVC_probabilities[0][0],2)}, "subjective": {round(SVC_probabilities[0][1],2)}<br/>The Logistic Regression model classified this text as: {LR_output}<br/>Probabilities for "objective": {round(LR_probabilities[0][0],2)}, "subjective": {round(LR_probabilities[0][1],2)}<br/>The Random Forest Classifier classified this text as: {RF_output}<br/>Probabilities for "objective": {round(RF_probabilities[0][0],2)}, "subjective": {round(RF_probabilities[0][1],2)}'))

# Note that the render_template method has a new parameter I named prediction_text. This parameter contains a message that will pop up after the user clicks on the predict button. This message is now in the back end. To send it to the front end, add the prediction_text variable inside the <body> of the index.html file.

if __name__ == "__main__":
    app.run()

# You can run the Flask application by running the py script in the Terminal (python <script_name>)