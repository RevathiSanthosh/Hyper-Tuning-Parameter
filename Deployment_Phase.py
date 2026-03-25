import pickle
import numpy as np

# Load trained model
model = pickle.load(open("final_model.pkl", "rb"))

# Example input
# Format: [age, sex, bmi, children, smoker, region_northwest, region_southeast, region_southwest]

input_data = np.array([[25, 1, 28.5, 2, 0, 1, 0, 0]])

# Prediction
prediction = model.predict(input_data)

print("Predicted Insurance Charges:", round(prediction[0], 2))