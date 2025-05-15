import os
import numpy as np
import pandas as pd
import joblib
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils import resample

MODEL_PATH = "models/"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

print("Loading and preprocessing data...")

data = pd.read_csv("dataset_diabetes/diabetic_data.csv")

admission_type_dict = {
    1: 'Emergency',
    2: 'Urgent',
    3: 'Elective',
    4: 'Newborn',
    5: 'Not Available',
    6: np.nan,
    7: 'Trauma Center',
    8: 'Not Mapped'
}
data['admission_type_id'] = data['admission_type_id'].map(admission_type_dict)
data['admission_type_id'] = data['admission_type_id'].fillna('Not Available')

discharge_disposition_dict={
    1:'Discharged to home',
    2:'Discharged/transferred to another short term hospital',
    3:'Discharged/transferred to SNF',
    4:'Discharged/transferred to ICF',
    5:'Discharged/transferred to another type of inpatient care institution',
    6:'Discharged/transferred to home with home health service',
    7:'Left AMA',
    8:'Discharged/transferred to home under care of Home IV provider',
    9:'Admitted as an inpatient to this hospital',
    10:'Neonate discharged to another hospital for neonatal aftercare',
    11:'Expired',
    12:'Still patient or expected to return for outpatient services',
    13:'Hospice / home',
    14:'Hospice / medical facility',
    15:'Discharged/transferred within this institution to Medicare approved swing bed',
    16:'Discharged/transferred/referred another institution for outpatient services',
    17:'Discharged/transferred/referred to this institution for outpatient services',
    18:np.nan,
    19:'Expired at home. Medicaid only, hospice.',
    20:'Expired in a medical facility. Medicaid only, hospice.',
    21:'Expired, place unknown. Medicaid only, hospice.',
    22:'Discharged/transferred to another rehab fac including rehab units of a hospital .',
    23:'Discharged/transferred to a long term care hospital.',
    24:'Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.',
    25:'Not Mapped',
    26:'Unknown/Invalid',
    30:'Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere',
    27:'Discharged/transferred to a federal health care facility.',
    28:'Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital',
    29:'Discharged/transferred to a Critical Access Hospital (CAH).',
}
data['discharge_disposition_id'] = data['discharge_disposition_id'].map(discharge_disposition_dict)
data['discharge_disposition_id'] = data['discharge_disposition_id'].fillna('Not Available')

admission_source_dict={
    1:'Physician Referral',
    2:'Clinic Referral',
    3:'HMO Referral',
    4:'Transfer from a hospital',
    5:'Transfer from a Skilled Nursing Facility (SNF)',
    6:'Transfer from another health care facility',
    7:'Emergency Room',
    8:'Court/Law Enforcement',
    9:'Not Available',
    10:'Transfer from critial access hospital',
    11:'Normal Delivery',
    12:'Premature Delivery',
    13:'Sick Baby',
    14:'Extramural Birth',
    15:'Not Available',
    17:np.nan,
    18:'Transfer From Another Home Health Agency',
    19:'Readmission to Same Home Health Agency',
    20:'Not Mapped',
    21:'Unknown/Invalid',
    22:'Transfer from hospital inpt/same fac reslt in a sep claim',
    23:'Born inside this hospital',
    24:'Born outside this hospital',
    25:'Transfer from Ambulatory Surgery Center',
    26:'Transfer from Hospice',
}
data['admission_source_id'] = data['admission_source_id'].map(admission_source_dict)
data['admission_source_id'] = data['admission_source_id'].fillna('Not Available')

data.replace('?', np.nan, inplace=True)

data.drop(columns=['weight', 'encounter_id', 'patient_nbr'], inplace=True)

data.dropna(subset=['diag_1'], inplace=True)

data['race'] = data['race'].fillna(data['race'].mode()[0])
data['A1Cresult'] = data['A1Cresult'].fillna('not measured')
data['max_glu_serum'] = data['max_glu_serum'].fillna('not measured')
data['payer_code'] = data['payer_code'].fillna('unknown')
data['medical_specialty'] = data['medical_specialty'].fillna('unknown')
data['diag_2'] = data['diag_2'].fillna("NO_SECONDARY_DX")
data['diag_3'] = data['diag_3'].fillna("NO_TERTIARY_DX")

data = data[data['gender'].isin(['Male', 'Female'])]

medication_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
                 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
                 'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin',
                 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
data['num_medications_taken'] = data[medication_cols].apply(lambda x: (x != 'No').sum(), axis=1)

def categorize_diagnosis(code):
    if pd.isna(code):
        return "Unknown"

    code = str(code).strip()

    if code.replace(".", "").isdigit():
        code = float(code)
        if 1 <= code <= 139:
            return "Infectious Diseases"
        elif 140 <= code <= 239:
            return "Cancer & Neoplasms"
        elif 240 <= code <= 279:
            return "Endocrine Disorders"
        elif 280 <= code <= 289:
            return "Blood Disorders"
        elif 290 <= code <= 319:
            return "Mental Health Disorders"
        elif 320 <= code <= 389:
            return "Nervous System Diseases"
        elif 390 <= code <= 459:
            return "Heart & Circulatory Conditions"
        elif 460 <= code <= 519:
            return "Respiratory Diseases"
        elif 520 <= code <= 579:
            return "Digestive System Diseases"
        elif 580 <= code <= 629:
            return "Kidney & Urinary Disorders"
        elif 630 <= code <= 679:
            return "Pregnancy-Related Conditions"
        elif 680 <= code <= 709:
            return "Skin Disorders"
        elif 710 <= code <= 739:
            return "Muscle & Bone Conditions"
        elif 740 <= code <= 759:
            return "Congenital Disorders"
        elif 760 <= code <= 779:
            return "Perinatal Conditions"
        elif 780 <= code <= 799:
            return "Symptoms & Non-Specific Conditions"
        elif 800 <= code <= 999:
            return "Injuries & Poisoning"
        else:
            return "Unknown ICD Code"

    elif code.startswith("V"):
        return "External Injury (Vehicle-related)"
    elif code.startswith("W"):
        return "External Injury (Falls, Accidents)"
    elif code.startswith("X"):
        return "External Injury (Poisoning, Assault)"
    elif code.startswith("Y"):
        return "External Injury (Other Causes)"
    else:
        return "Unknown"

data["diag_category"] = data["diag_1"].apply(categorize_diagnosis)

data['readmitted_binary'] = (data['readmitted'] == '<30').astype(int)

print(f"Class distribution before balancing: {data['readmitted_binary'].value_counts()}")

if len(data['readmitted_binary'].unique()) < 2:
    print("WARNING: Only one class found in the target variable.")
    print("Adding synthetic examples of the missing class...")
    
    majority_class = data['readmitted_binary'].iloc[0]
    minority_class = 1 if majority_class == 0 else 0
    
    synthetic_examples_count = int(len(data) * 0.1)
    print(f"Adding {synthetic_examples_count} synthetic examples of class {minority_class}")
    
    synthetic_examples = []
    for _ in range(synthetic_examples_count):
        random_idx = random.randint(0, len(data) - 1)
        synthetic_row = data.iloc[random_idx].copy()
        
        synthetic_row['readmitted_binary'] = minority_class
        synthetic_row['readmitted'] = '<30' if minority_class == 1 else '>30'
        
        if minority_class == 1:
            synthetic_row['number_inpatient'] = min(10, synthetic_row['number_inpatient'] + 1)
            synthetic_row['number_emergency'] = min(10, synthetic_row['number_emergency'] + 1)
            synthetic_row['time_in_hospital'] = min(30, synthetic_row['time_in_hospital'] + 2)
            synthetic_row['number_diagnoses'] = min(10, synthetic_row['number_diagnoses'] + 1)
        
        synthetic_examples.append(synthetic_row)
    
    synthetic_df = pd.DataFrame(synthetic_examples)
    
    data = pd.concat([data, synthetic_df], ignore_index=True)
    
    print(f"Class distribution after adding synthetic examples: {data['readmitted_binary'].value_counts()}")

categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

if 'age' in categorical_cols:
    categorical_cols.remove('age')

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

map_age = {
    '[0-10)': 0,
    '[10-20)': 1,
    '[20-30)': 2,
    '[30-40)': 3,
    '[40-50)': 4,
    '[50-60)': 5,
    '[60-70)': 6,
    '[70-80)': 7,
    '[80-90)': 8,
    '[90-100)': 9
}
data['age'] = data['age'].map(map_age)

scaler = StandardScaler()
numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
               'num_medications', 'number_outpatient', 'number_emergency',
               'number_inpatient', 'number_diagnoses', 'num_medications_taken']
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

mask = (data[numeric_cols] > -3) & (data[numeric_cols] < 3)
data = data[mask.all(axis=1)]

X = data.drop(columns=['readmitted', 'readmitted_binary'])
y = data['readmitted_binary']

print(f"Final class distribution: {y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

joblib.dump(list(X.columns), os.path.join(MODEL_PATH, 'column_names.pkl'))
joblib.dump(encoders, os.path.join(MODEL_PATH, 'label_encoders.pkl'))
joblib.dump(scaler, os.path.join(MODEL_PATH, 'scaler.pkl'))

diagnosis_categories = {
    'Infectious Diseases': (1, 139),
    'Cancer & Neoplasms': (140, 239),
    'Endocrine Disorders': (240, 279),
    'Blood Disorders': (280, 289),
    'Mental Health Disorders': (290, 319),
    'Nervous System Diseases': (320, 389),
    'Heart & Circulatory Conditions': (390, 459),
    'Respiratory Diseases': (460, 519),
    'Digestive System Diseases': (520, 579),
    'Kidney & Urinary Disorders': (580, 629),
    'Pregnancy-Related Conditions': (630, 679),
    'Skin Disorders': (680, 709),
    'Muscle & Bone Conditions': (710, 739),
    'Congenital Disorders': (740, 759),
    'Perinatal Conditions': (760, 779),
    'Symptoms & Non-Specific Conditions': (780, 799),
    'Injuries & Poisoning': (800, 999)
}
joblib.dump(diagnosis_categories, os.path.join(MODEL_PATH, 'diagnosis_categories.pkl'))

models = {
    'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'svm': SVC(probability=True, class_weight='balanced', kernel='linear', max_iter=1000, tol=1e-3),
    'random_forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    'knn': KNeighborsClassifier(n_neighbors=5)
}

print("Training and evaluating models...")
best_auc = 0
best_model_name = None

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
    
    print(f"{name.capitalize()} Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    if y_pred_proba is not None:
        print(f"  ROC AUC: {auc:.4f}")
    
    joblib.dump(model, os.path.join(MODEL_PATH, f'{name}_model.pkl'))
    
    if auc > best_auc:
        best_auc = auc
        best_model_name = name

print(f"\nBest performing model: {best_model_name} (AUC: {best_auc:.4f})")

if hasattr(models['random_forest'], 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': models['random_forest'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    joblib.dump(feature_importance, os.path.join(MODEL_PATH, 'feature_importance.pkl'))

with open(os.path.join(MODEL_PATH, 'best_model.txt'), 'w') as f:
    f.write(best_model_name)

print("Training complete. All models saved to the 'models' directory.") 