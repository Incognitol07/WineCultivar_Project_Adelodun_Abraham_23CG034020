import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load Model
MODEL_PATH = "model/wine_cultivar_model.pkl"
model = None


def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        print("Model file not found. Ensure model is trained.")


load_model()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        if not model:
            load_model()

        if model:
            try:
                # Extract features from form
                # Features: alcohol, flavanoids, color_intensity, hue, od280/od315_of_diluted_wines, proline
                features = [
                    float(request.form["alcohol"]),
                    float(request.form["flavanoids"]),
                    float(request.form["color_intensity"]),
                    float(request.form["hue"]),
                    float(request.form["od280/od315_of_diluted_wines"]),
                    float(request.form["proline"]),
                ]

                # Create DataFrame for prediction (to match feature names if needed by pipeline,
                # though usually numpy array works if pipeline doesn't strictly check names.
                # Better to use DataFrame with correct columns to be safe with some transformers)
                feature_names = [
                    "alcohol",
                    "flavanoids",
                    "color_intensity",
                    "hue",
                    "od280/od315_of_diluted_wines",
                    "proline",
                ]
                input_df = pd.DataFrame([features], columns=feature_names)

                # Predict
                pred_class = model.predict(input_df)[0]

                # Map class to name (Dataset target_names: class_0, class_1, class_2)
                # Usually wine dataset has classes 0, 1, 2.
                # Let's give them nice names if possible, but generic "Cultivar X" is compliant.
                prediction = f"Cultivar {pred_class + 1}"  # converting 0,1,2 to 1,2,3

            except Exception as e:
                prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
