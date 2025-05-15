# Hospital Readmission Prediction System

This application predicts the likelihood of a diabetic patient being readmitted to the hospital within 30 days based on their medical history, treatment information, and demographic details.

## Project Overview

The system uses machine learning models (when available) or a rule-based approach to predict patient readmission risk. The project includes:

1. Data exploration and preprocessing capabilities (in train_models.py)
2. Model training functionality (in train_models.py)
3. A graphical user interface for making predictions (working_gui.py)

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Required Python packages (install using `pip install -r requirements.txt`):
  - PyQt5
  - numpy
  - pandas
  - joblib (for model loading)

### Dataset

The application references the diabetes dataset located in the `dataset_diabetes` directory:
- `diabetic_data.csv`: The main dataset with patient records
- `preprocessed_data.csv`: Preprocessed version of the data
- `IDs_mapping.csv`: Mapping of patient identifiers

## Usage

### Running the Application

To run the application, use the following command:

```
python working_gui.py
```

This will open the graphical interface where you can:
1. Enter patient information
2. Click "Predict Readmission Risk" to get a prediction
3. View the prediction result and confidence score
4. See a visual representation of the risk level with a color-coded progress bar
5. View the key risk factors that contributed to the prediction

### Training Models

To train machine learning models for prediction, run:

```
python train_models.py
```

This will preprocess the data, train several models (logistic regression, SVM, random forest, and k-nearest neighbors), and save the best performing model along with necessary preprocessors to the `models` directory.

## Features

- **Machine Learning Integration**: Uses trained ML models when available, with fallback to rule-based prediction
- **Patient Information Entry**: Input form for demographics, hospital information, and test results
- **Risk Prediction**: Calculation of readmission probability based on patient data
- **Visualization**: Color-coded progress bar showing risk level
- **Factor Analysis**: Display of key factors contributing to the readmission risk

## Project Structure

- `working_gui.py`: Main GUI application (enhanced with ML model integration)
- `train_models.py`: Script for training machine learning models
- `project.py`: Data exploration and analysis of the dataset
- `requirements.txt`: List of required Python packages
- `dataset_diabetes/`: Directory containing the diabetes dataset
- `models/`: Directory containing trained models and preprocessors 