from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

def train_model(csv_path='tested.csv'):
    """
    Loads CSV, does the preprocessing you posted, trains and returns the model.
    This function is executed once when the app starts (so the model is retrained each run).
    """
    df = pd.read_csv(csv_path)

    # your preprocessing (kept intact, with small safety tweaks)
    df['Upd_Age']  = df['Age'].fillna(df['Age'].mean()); df.drop('Age', axis=1, inplace=True)
    df['Upd_Fare'] = df['Fare'].fillna(df['Fare'].median()); df.drop('Fare', axis=1, inplace=True)
    # drop Cabin if it exists
    if 'Cabin' in df.columns:
        df.drop('Cabin', axis=1, inplace=True)

    # map categorical
    df['Sex']      = df['Sex'].map({'male':0, 'female':1})
    df['Embarked'] = df['Embarked'].map({'Q':1, 'S':2, 'C':3})

    # features + label
    feature_cols = ['Sex','Parch','SibSp','Upd_Age','Upd_Fare','Pclass','Embarked']
    X = df[feature_cols]
    y = df['Survived']

    # increase max_iter to avoid convergence warnings on some datasets
    model = LogisticRegression(random_state=16, max_iter=200)
    model.fit(X, y)

    return model, feature_cols

# Train on startup (retrained every time you run `python app.py`)
model, feature_cols = train_model('tested.csv')


@app.route('/')
def home():
    # renders form (templates/home.html)
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # get form values and convert types to match training features order
    try:
        pclass   = int(request.form['Pclass'])
        sex      = int(request.form['Sex'])
        age      = float(request.form['Age'])
        sibsp    = int(request.form['SibSp'])
        parch    = int(request.form['Parch'])
        fare     = float(request.form['Fare'])
        embarked = int(request.form['Embarked'])
    except Exception as e:
        return f"Invalid input: {e}", 400

    # Construct input in the same order as feature_cols:
    features = [[sex, parch, sibsp, age, fare, pclass, embarked]]

    pred = model.predict(features)[0]
    result = "Survived" if pred == 1 else "Not Survived"

    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
