# Titanic_Passenger_Survival_Prediction
A simple Flask web application that trains a Logistic Regression model on the Titanic dataset and lets users predict whether a passenger would survive by filling  web form. The app retrains the model every time the Flask process starts (so the model is always trained from the CSV on startup).
# Project structure
titanic-flask-app/
├─ app.py                # Main Flask application (trains model at startup)
├─ requirements.txt      # Python dependencies
├─ tested.csv            # Titanic dataset used for training (place here)
├─ templates/
│  ├─ home.html          # Input form page
│  └─ result.html        # Prediction result page
└─ static/
   └─ styles.css         # Small custom CSS for theming
