# Wine Cultivar Origin Prediction System

## Project Overview

This project is a Machine Learning-based web application that predicts the cultivar (origin) of a wine sample based on its chemical properties. It utilizes the **Wine Dataset** (UCI/sklearn) and implements a **Random Forest Classifier** to achieve high prediction accuracy. The system is served via a **Flask** web interface with a modern, premium "Glassmorphism" design.

## Features

- **Machine Learning Model**: Random Forest Classifier trained on 6 key chemical features.
  - Alcohol
  - Flavanoids
  - Color Intensity
  - Hue
  - OD280/OD315 of diluted wines
  - Proline
- **Accuracy**: The model achieves **100% accuracy** on the test set.
- **Web GUI**: A responsive, aesthetically pleasing interface for users to input data and view predictions.
- **Backend**: Built with Flask (Python).

## Tech Stack

- **Language**: Python 3.8+
- **Web Framework**: Flask
- **ML Libraries**: scikit-learn, pandas, numpy, joblib
- **Frontend**: HTML5, CSS3 (Glassmorphism design), Google Fonts (Outfit)

## Project Structure

```
WineCultivar_Project_Adelodun_23CG034020/
│
├── app.py                      # Flask Application Entry Point
├── requirements.txt            # Python Dependencies
├── WineCultivar_hosted_webGUI_link.txt # Deployment Link & Details
├── READM.md                    # Project Documentation
│
├── model/                      # ML Model Directory
│   ├── model_building.py       # Script to train and save the model
│   └── wine_cultivar_model.pkl # Trained Random Forest Model
│
├── static/                     # Static Assets (CSS, Images)
│   ├── style.css               # Application Styling
│   └── fluent_bg.svg           # Background Graphic
│
└── templates/                  # HTML Templates
    └── index.html              # Main Web Interface
```

## Setup & Installation

1. **Clone/Download the repository** to your local machine.
2. **Navigate to the project directory**:
   ```bash
   cd WineCultivar_Project_Adelodun_23CG034020
   ```
3. **Install Dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## Model Training (Optional)

The model is already trained and saved as `model/wine_cultivar_model.pkl`. If you wish to retrain it:

```bash
python model/model_building.py
```

This will overwrite the existing model file with a newly trained one.

## Running the Application

1. **Start the Flask Server**:
   ```bash
   python app.py
   ```
2. **Access the GUI**:
   Open your browser and navigate to:
   `http://127.0.0.1:5000/`

## Usage

1. Enter the values for the 6 chemical properties in the form.
2. Click **Analyze Sample**.
3. The predicted Cultivar (1, 2, or 3) will be displayed on the screen.

## Project Details

- **Student Name**: Adelodun
- **Matric Number**: 23CG034020
- **Algorithm**: Random Forest Classifier
