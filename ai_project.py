import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

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
print(data[['admission_type_id','discharge_disposition_id','admission_source_id']])

# Task 2: Handle missing values

# Replace '?' with NaN values
data.replace('?',np.nan,inplace=True)

# Display count of null values in each column
print(data.isnull().sum())

# Drop weight column (98% missing values)
data.drop(columns=['weight'],inplace=True)

# Check race distribution
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
print(data.isnull().sum())

# Task 3: Encode categorical features (Label Encoding)
le = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
  data[column] = le.fit_transform(data[column])

# --- FADY'S TASKS ---
# Task 1: One-hot encoding for categorical variables
categorical_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id', 
                    'race', 'gender', 'age', 'payer_code', 'medical_specialty', 
                    'A1Cresult', 'max_glu_serum', 'change', 'diabetesMed', 'readmitted']
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_cats = onehot_encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_cats, columns=onehot_encoder.get_feature_names_out(categorical_cols))

# Drop original columns and concatenate one-hot encoded columns
data = data.drop(columns=categorical_cols)
data = pd.concat([data, encoded_df], axis=1)

# Task 2: Remove outliers using IQR method
numeric_cols = data.select_dtypes(include=['int64','float64']).columns
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data = data[(data[col] >= lower) & (data[col] <= upper)]

# Task 3: Standardize numerical columns
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# --- END OF PREPROCESSING ---
# Data is now ready for EDA and model development

# Save preprocessed data now is ready for use
data.to_csv("dataset_diabetes/preprocessed_data.csv", index=False)