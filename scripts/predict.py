
import sys
from joblib import load

# Load the model
model = load('../model/sentiment_model.joblib')

# Predict the sentiment
reviews = [sys.argv[1]]
predictions = model.predict(reviews)
print(predictions[0])
