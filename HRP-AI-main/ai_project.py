import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- DATA LOADING ---
# Load diabetes dataset
data = pd.read_csv("dataset_diabetes/diabetic_data.csv")

# --- ABDULRAHMAN'S TASKS ---
# Task 1: Decode IDs using mapping file

# Decode admission_type_id
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

# Decode discharge_disposition_id
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
data['discharge_disposition_id']  = data['discharge_disposition_id'].map(discharge_disposition_dict)

# Decode admission_source_id
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

# Display decoded categorical fields
print(data[['admission_type_id','discharge_disposition_id','admission_source_id']].head())

# Task 2: Handle missing values

# Replace '?' with NaN values
data.replace('?',np.nan,inplace=True)

# Display count of null values in each column
print("Null values in each column:")
print(data.isnull().sum())

# Drop weight column (98% missing values)
data.drop(columns=['weight'],inplace=True)

# Check race distribution
print("Race distribution:")
print(data['race'].value_counts(normalize=True))

# Handle missing values for each column
data['race'] = data['race'].fillna(data['race'].mode()[0])  # Fill with mode
data['A1Cresult'] = data['A1Cresult'].fillna('not measured')  # Medical test results
data['max_glu_serum'] = data['max_glu_serum'].fillna('not measured')
data['payer_code'] = data['payer_code'].fillna('unknown')
data['medical_specialty'] = data['medical_specialty'].fillna('unknown')
data['diag_2'] = data['diag_2'].fillna("NO_SECONDARY_DX")  # Secondary diagnosis
data['diag_3'] = data['diag_3'].fillna("NO_TERTIARY_DX")  # Tertiary diagnosis
data['admission_type_id'] = data['admission_type_id'].fillna('Not Available')
data['discharge_disposition_id'] = data['discharge_disposition_id'].fillna('Not Available')
data['admission_source_id'] = data['admission_source_id'].fillna('Not Available')

# Remove rows with missing primary diagnosis (critical field)
data.dropna(subset=['diag_1'], inplace=True)

# Verify all nulls are handled
print("\nAfter handling nulls:")
print(data.isnull().sum().sum())

print(f"Data shape after handling missing values: {data.shape}")

# --- FADY'S TASKS ---
print("\nTarget variable distribution before transformation:")
print(data['readmitted'].value_counts())

# Map the target variable 'readmitted'
readmitted_mapping = {
    'NO': 0,
    '>30': 0,
    '<30': 1
}
if '<30' in data['readmitted'].values:
    data['readmitted'] = data['readmitted'].map(readmitted_mapping)
    print("Converted readmitted to binary: 1 for <30 days, 0 for others")
    print(data['readmitted'].value_counts())

# Task 3: Encode categorical features using Label Encoding
le = LabelEncoder()
categorical_cols = data.select_dtypes(include=['object']).columns
print(f"\nApplying Label Encoding to {len(categorical_cols)} categorical columns")

for column in categorical_cols:
    if column != 'readmitted':
        data[column] = le.fit_transform(data[column])
        print(f"Encoded {column}")

# Task 2: Remove outliers using IQR method
numeric_cols_for_outlier_removal = ['num_lab_procedures', 'num_procedures', 'num_medications',
                                   'number_outpatient', 'number_emergency', 
                                   'number_inpatient', 'number_diagnoses']

print(f"Data shape before outlier removal: {data.shape}")
original_count = len(data)

for col in numeric_cols_for_outlier_removal:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3.0 * IQR
    upper = Q3 + 3.0 * IQR
    data = data[(data[col] >= lower) & (data[col] <= upper)]

removed_pct = (original_count - len(data)) / original_count * 100
print(f"Removed {removed_pct:.2f}% of rows as outliers")
print(f"Data shape after outlier removal: {data.shape}")

# Task 3: Standardize numerical columns
numeric_cols_to_standardize = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
binary_cols = []
for col in numeric_cols_to_standardize:
    if data[col].nunique() <= 2:
        binary_cols.append(col)

numeric_cols = [col for col in numeric_cols_to_standardize if col not in binary_cols]

if numeric_cols_to_standardize:
    print(f"Standardizing {len(numeric_cols_to_standardize)} numerical columns")
    print(f"Examples: {numeric_cols_to_standardize[:5]}")
    
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# --- END OF PREPROCESSING ---
print(f"\nFinal data shape: {data.shape}")
# Data is now ready for EDA and model development

# Save preprocessed data now is ready for use
data.to_csv("dataset_diabetes/preprocessed_data.csv", index=False)
print("Saved preprocessed data to dataset_diabetes/preprocessed_data.csv")