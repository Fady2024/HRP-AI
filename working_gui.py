import sys
import os
import numpy as np
import pandas as pd
import joblib
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QComboBox, QSpinBox, QPushButton, 
                            QGroupBox, QFormLayout, QScrollArea, 
                            QMessageBox, QSplitter, QProgressBar, 
                            QTabWidget, QTableWidget, QTableWidgetItem,
                            QCheckBox, QRadioButton, QButtonGroup, QHeaderView)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

print("Starting hospital readmission prediction application...")

MODEL_PATH = "models/"

class HospitalReadmissionGUI(QWidget):
    def __init__(self):
        super().__init__()
        print("Initializing GUI...")
        
        self.setWindowTitle("Hospital Readmission Prediction System")
        self.setMinimumSize(1200, 800)

        self.load_models()

        main_layout = QHBoxLayout(self)

        input_widget = QWidget()
        self.input_layout = QVBoxLayout(input_widget)
        self.create_input_form()

        results_widget = QWidget()
        self.results_layout = QVBoxLayout(results_widget)
        self.create_results_section()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(input_widget)
        splitter.addWidget(results_widget)
        splitter.setSizes([500, 700])
        
        main_layout.addWidget(splitter)
        
        print("GUI initialization complete")
    
    def load_models(self):
        """Load the trained models and preprocessors if available"""
        try:

            if os.path.exists(os.path.join(MODEL_PATH, 'best_model.txt')):
                with open(os.path.join(MODEL_PATH, 'best_model.txt'), 'r') as f:
                    self.best_model_name = f.read().strip()
                
                self.model = joblib.load(os.path.join(MODEL_PATH, f'{self.best_model_name}_model.pkl'))
                self.encoders = joblib.load(os.path.join(MODEL_PATH, 'label_encoders.pkl'))
                self.scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
                self.column_names = joblib.load(os.path.join(MODEL_PATH, 'column_names.pkl'))
                self.diagnosis_categories = joblib.load(os.path.join(MODEL_PATH, 'diagnosis_categories.pkl'))

                if os.path.exists(os.path.join(MODEL_PATH, 'feature_importance.pkl')):
                    self.feature_importance = joblib.load(os.path.join(MODEL_PATH, 'feature_importance.pkl'))
                
                print(f"Models loaded successfully. Using {self.best_model_name} model.")
                self.use_ml_model = True
            else:
                print("No trained models found. Using rule-based approach.")
                self.use_ml_model = False

                self.diagnosis_categories = {
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
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            print("Falling back to rule-based approach.")
            self.use_ml_model = False
    
    def create_input_form(self):
        """Create the input form for patient data"""
        print("Creating input form...")
        form_title = QLabel("Patient Information")
        form_title.setFont(QFont("Arial", 14, QFont.Bold))
        self.input_layout.addWidget(form_title)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        input_tabs = QTabWidget()

        demographics_tab = QWidget()
        demographics_layout = QVBoxLayout(demographics_tab)

        demographics_group = QGroupBox("Demographics")
        demographics_form = QFormLayout()

        self.age_combo = QComboBox()
        self.age_groups = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', 
                          '[40-50)', '[50-60)', '[60-70)', '[70-80)', 
                          '[80-90)', '[90-100)']
        self.age_combo.addItems(self.age_groups)
        self.age_combo.setCurrentIndex(6)  
        demographics_form.addRow("Age Group:", self.age_combo)

        self.gender_combo = QComboBox()
        self.gender_combo.addItems(['Female', 'Male'])
        demographics_form.addRow("Gender:", self.gender_combo)

        self.race_combo = QComboBox()
        self.race_combo.addItems(['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other'])
        demographics_form.addRow("Race:", self.race_combo)
        
        demographics_group.setLayout(demographics_form)
        demographics_layout.addWidget(demographics_group)

        input_tabs.addTab(demographics_tab, "Demographics")

        hospital_tab = QWidget()
        hospital_layout = QVBoxLayout(hospital_tab)

        hospital_group = QGroupBox("Hospital Information")
        hospital_form = QFormLayout()

        self.admission_type_combo = QComboBox()
        admission_types = ['Emergency', 'Urgent', 'Elective', 'Newborn', 
                          'Trauma Center', 'Not Available', 'Not Mapped']
        self.admission_type_combo.addItems(admission_types)
        self.admission_type_combo.setCurrentIndex(0)  
        hospital_form.addRow("Admission Type:", self.admission_type_combo)

        self.admission_source_combo = QComboBox()
        admission_sources = ['Physician Referral', 'Clinic Referral', 'HMO Referral', 
                            'Transfer from a hospital', 'Transfer from a SNF', 
                            'Transfer from another health care facility', 'Emergency Room', 
                            'Court/Law Enforcement', 'Not Available']
        self.admission_source_combo.addItems(admission_sources)
        self.admission_source_combo.setCurrentIndex(6)  
        hospital_form.addRow("Admission Source:", self.admission_source_combo)

        self.time_in_hospital = QSpinBox()
        self.time_in_hospital.setRange(1, 30)
        self.time_in_hospital.setValue(5)
        hospital_form.addRow("Time in Hospital (days):", self.time_in_hospital)

        self.num_lab_procedures = QSpinBox()
        self.num_lab_procedures.setRange(1, 150)
        self.num_lab_procedures.setValue(50)
        hospital_form.addRow("Number of Lab Procedures:", self.num_lab_procedures)

        self.num_procedures = QSpinBox()
        self.num_procedures.setRange(0, 10)
        self.num_procedures.setValue(1)
        hospital_form.addRow("Number of Procedures:", self.num_procedures)

        self.num_medications = QSpinBox()
        self.num_medications.setRange(1, 30)
        self.num_medications.setValue(15)
        hospital_form.addRow("Number of Medications:", self.num_medications)
        
        hospital_group.setLayout(hospital_form)
        hospital_layout.addWidget(hospital_group)

        visits_group = QGroupBox("Previous Visits")
        visits_form = QFormLayout()

        self.number_outpatient = QSpinBox()
        self.number_outpatient.setRange(0, 10)
        self.number_outpatient.setValue(0)
        visits_form.addRow("Number of Outpatient Visits:", self.number_outpatient)

        self.number_emergency = QSpinBox()
        self.number_emergency.setRange(0, 10)
        self.number_emergency.setValue(0)
        visits_form.addRow("Number of Emergency Visits:", self.number_emergency)

        self.number_inpatient = QSpinBox()
        self.number_inpatient.setRange(0, 10)
        self.number_inpatient.setValue(0)
        visits_form.addRow("Number of Inpatient Visits:", self.number_inpatient)
        
        visits_group.setLayout(visits_form)
        hospital_layout.addWidget(visits_group)

        input_tabs.addTab(hospital_tab, "Hospital Information")

        diagnosis_tab = QWidget()
        diagnosis_layout = QVBoxLayout(diagnosis_tab)

        diagnosis_group = QGroupBox("Diagnosis Information")
        diagnosis_form = QFormLayout()

        self.number_diagnoses = QSpinBox()
        self.number_diagnoses.setRange(1, 10)
        self.number_diagnoses.setValue(5)
        diagnosis_form.addRow("Number of Diagnoses:", self.number_diagnoses)

        self.diagnosis_combo = QComboBox()
        self.disease_categories = list(self.diagnosis_categories.keys())
        self.diagnosis_combo.addItems(self.disease_categories)
        self.diagnosis_combo.setCurrentIndex(self.disease_categories.index("Heart & Circulatory Conditions") if "Heart & Circulatory Conditions" in self.disease_categories else 0)
        diagnosis_form.addRow("Primary Diagnosis:", self.diagnosis_combo)

        self.medical_specialty_combo = QComboBox()
        specialties = ['InternalMedicine', 'Cardiology', 'Surgery', 'Nephrology', 
                      'Emergency/Trauma', 'Endocrinology', 'Family/GeneralPractice', 
                      'Other', 'unknown']
        self.medical_specialty_combo.addItems(specialties)
        self.medical_specialty_combo.setCurrentIndex(0)
        diagnosis_form.addRow("Medical Specialty:", self.medical_specialty_combo)
        
        diagnosis_group.setLayout(diagnosis_form)
        diagnosis_layout.addWidget(diagnosis_group)

        tests_group = QGroupBox("Lab Tests")
        tests_form = QFormLayout()

        self.a1c_combo = QComboBox()
        self.a1c_combo.addItems(['Norm', '>7', '>8', 'not measured'])
        tests_form.addRow("A1C Test Result:", self.a1c_combo)

        self.glu_serum_combo = QComboBox()
        self.glu_serum_combo.addItems(['Norm', '>200', '>300', 'not measured'])
        tests_form.addRow("Glucose Serum Test:", self.glu_serum_combo)
        
        tests_group.setLayout(tests_form)
        diagnosis_layout.addWidget(tests_group)

        input_tabs.addTab(diagnosis_tab, "Diagnosis & Tests")

        meds_tab = QWidget()
        meds_layout = QVBoxLayout(meds_tab)

        meds_group = QGroupBox("Medication Information")
        meds_form = QFormLayout()

        self.diabetes_med_combo = QComboBox()
        self.diabetes_med_combo.addItems(['Yes', 'No'])
        meds_form.addRow("Diabetes Medication:", self.diabetes_med_combo)

        self.med_change_combo = QComboBox()
        self.med_change_combo.addItems(['No', 'Ch'])
        meds_form.addRow("Medication Change:", self.med_change_combo)

        self.insulin_combo = QComboBox()
        self.insulin_combo.addItems(['No', 'Up', 'Down', 'Steady'])
        meds_form.addRow("Insulin:", self.insulin_combo)
        
        meds_group.setLayout(meds_form)
        meds_layout.addWidget(meds_group)

        specific_meds_group = QGroupBox("Specific Medications")
        specific_meds_layout = QVBoxLayout()

        self.medication_checks = {}
        medication_subgroups = {
            "Diabetes Medications": ['metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone']
        }
        
        for group_name, meds in medication_subgroups.items():
            group_box = QGroupBox(group_name)
            group_layout = QVBoxLayout()
            
            for med in meds:
                check = QCheckBox(med.capitalize())
                self.medication_checks[med] = check
                group_layout.addWidget(check)
            
            group_box.setLayout(group_layout)
            specific_meds_layout.addWidget(group_box)
        
        specific_meds_group.setLayout(specific_meds_layout)
        meds_layout.addWidget(specific_meds_group)

        input_tabs.addTab(meds_tab, "Medications")

        scroll_layout.addWidget(input_tabs)

        self.predict_button = QPushButton("Predict Readmission Risk")
        self.predict_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.predict_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        self.predict_button.clicked.connect(self.predict_readmission)
        scroll_layout.addWidget(self.predict_button)
        
        scroll_area.setWidget(scroll_content)
        self.input_layout.addWidget(scroll_area)
        print("Input form created")
    
    def create_results_section(self):
        """Create the results display section"""
        print("Creating results section...")
        results_title = QLabel("Prediction Results")
        results_title.setFont(QFont("Arial", 14, QFont.Bold))
        self.results_layout.addWidget(results_title)

        self.results_tabs = QTabWidget()

        prediction_tab = QWidget()
        prediction_layout = QVBoxLayout(prediction_tab)

        self.prediction_label = QLabel("No prediction available yet.")
        self.prediction_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.prediction_label.setAlignment(Qt.AlignCenter)
        prediction_layout.addWidget(self.prediction_label)

        self.confidence_label = QLabel("")
        self.confidence_label.setFont(QFont("Arial", 14))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        prediction_layout.addWidget(self.confidence_label)

        risk_group = QGroupBox("Risk Level")
        risk_layout = QVBoxLayout()
        
        self.risk_progress = QProgressBar()
        self.risk_progress.setRange(0, 100)
        self.risk_progress.setValue(0)
        self.risk_progress.setTextVisible(True)
        self.risk_progress.setFormat("%p%")
        self.risk_progress.setMinimumHeight(30)
        risk_layout.addWidget(self.risk_progress)
        
        risk_group.setLayout(risk_layout)
        prediction_layout.addWidget(risk_group)

        self.risk_interpretation = QLabel("Readmission risk interpretation will appear here.")
        self.risk_interpretation.setWordWrap(True)
        self.risk_interpretation.setAlignment(Qt.AlignCenter)
        prediction_layout.addWidget(self.risk_interpretation)

        self.results_tabs.addTab(prediction_tab, "Prediction")

        importance_tab = QWidget()
        importance_layout = QVBoxLayout(importance_tab)
        
        importance_label = QLabel("Key Factors Contributing to Prediction")
        importance_label.setFont(QFont("Arial", 12, QFont.Bold))
        importance_label.setAlignment(Qt.AlignCenter)
        importance_layout.addWidget(importance_label)

        self.feature_table = QTableWidget(10, 2)
        self.feature_table.setHorizontalHeaderLabels(["Feature", "Importance"])
        self.feature_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.feature_table.setEditTriggers(QTableWidget.NoEditTriggers)
        importance_layout.addWidget(self.feature_table)

        self.results_tabs.addTab(importance_tab, "Feature Importance")

        risk_factors_tab = QWidget()
        risk_factors_layout = QVBoxLayout(risk_factors_tab)
        
        risk_factors_label = QLabel("Patient Risk Factors Analysis")
        risk_factors_label.setFont(QFont("Arial", 12, QFont.Bold))
        risk_factors_label.setAlignment(Qt.AlignCenter)
        risk_factors_layout.addWidget(risk_factors_label)
        
        self.risk_factors_text = QLabel("Risk factors analysis will appear here after prediction.")
        self.risk_factors_text.setWordWrap(True)
        self.risk_factors_text.setStyleSheet("background-color: #f8f9fa; padding: 10px; border-radius: 5px;")
        risk_factors_layout.addWidget(self.risk_factors_text)

        self.risk_factors_table = QTableWidget(5, 2)
        self.risk_factors_table.setHorizontalHeaderLabels(["Risk Factor", "Status"])
        self.risk_factors_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.risk_factors_table.setEditTriggers(QTableWidget.NoEditTriggers)
        risk_factors_layout.addWidget(self.risk_factors_table)

        self.results_tabs.addTab(risk_factors_tab, "Risk Factors")

        self.results_layout.addWidget(self.results_tabs)
        
        print("Results section created")
    
    def predict_readmission(self):
        """Collect input data and make prediction"""
        try:
            print("Making prediction...")

            age_group = self.age_combo.currentText()
            gender = self.gender_combo.currentText()
            race = self.race_combo.currentText()

            admission_type = self.admission_type_combo.currentText()
            admission_source = self.admission_source_combo.currentText()
            time_in_hospital = self.time_in_hospital.value()
            num_lab_procedures = self.num_lab_procedures.value()
            num_procedures = self.num_procedures.value()
            num_medications = self.num_medications.value()

            number_outpatient = self.number_outpatient.value()
            number_emergency = self.number_emergency.value()
            number_inpatient = self.number_inpatient.value()

            number_diagnoses = self.number_diagnoses.value()
            diagnosis_category = self.diagnosis_combo.currentText()
            medical_specialty = self.medical_specialty_combo.currentText()
            a1c_result = self.a1c_combo.currentText()
            glu_serum = self.glu_serum_combo.currentText()

            diabetes_med = self.diabetes_med_combo.currentText()
            med_change = self.med_change_combo.currentText()
            insulin = self.insulin_combo.currentText()

            medications_taken = {}
            for med_name, checkbox in self.medication_checks.items():
                medications_taken[med_name] = "Up" if checkbox.isChecked() else "No"

            checked_medications = sum(1 for checkbox in self.medication_checks.values() if checkbox.isChecked())

            force_high_risk = False
            extreme_risk_count = 0

            if number_inpatient >= 5:
                extreme_risk_count += 1
            if number_emergency >= 5:
                extreme_risk_count += 1
            if self.age_groups.index(age_group) >= 8:  
                extreme_risk_count += 1
            high_risk_diagnoses = ['Heart & Circulatory Conditions', 'Respiratory Diseases', 
                               'Kidney & Urinary Disorders', 'Endocrine Disorders']
            if diagnosis_category in high_risk_diagnoses:
                extreme_risk_count += 1
            if a1c_result == '>8' or glu_serum == '>300':
                extreme_risk_count += 1
            if time_in_hospital >= 20:
                extreme_risk_count += 1
            if med_change == 'Ch' and insulin == 'Up' and diabetes_med == 'Yes':
                extreme_risk_count += 1

            if extreme_risk_count >= 3:
                force_high_risk = True
                print(f"Forcing high risk due to {extreme_risk_count} extreme risk factors")

            if self.use_ml_model and hasattr(self, 'model') and not force_high_risk:

                probability, features_importance = self.predict_with_model(
                    age_group, gender, race, admission_type, admission_source, 
                    time_in_hospital, num_lab_procedures, num_procedures, num_medications,
                    number_outpatient, number_emergency, number_inpatient,
                    number_diagnoses, diagnosis_category, medical_specialty,
                    a1c_result, glu_serum, diabetes_med, med_change, insulin,
                    medications_taken
                )
            else:

                probability = self.predict_with_rules(
                    age_group, gender, time_in_hospital, num_procedures,
                    num_medications, number_emergency, number_inpatient,
                    number_diagnoses, diagnosis_category, a1c_result, glu_serum,
                    insulin
                )
                features_importance = None

            if force_high_risk:
                probability = max(0.75, probability)

            prediction = "High Risk" if probability > 0.5 else "Low Risk"
            confidence = probability if prediction == "High Risk" else 1 - probability

            self.prediction_label.setText(f"Readmission Prediction: {prediction}")
            self.confidence_label.setText(f"Confidence: {confidence:.2%}")

            self.risk_progress.setValue(int(probability * 100))

            if probability < 0.3:
                self.risk_progress.setStyleSheet("QProgressBar::chunk { background-color: green; }")
                risk_level = "Low Risk"
            elif probability < 0.6:
                self.risk_progress.setStyleSheet("QProgressBar::chunk { background-color: yellow; }")
                risk_level = "Moderate Risk"
            else:
                self.risk_progress.setStyleSheet("QProgressBar::chunk { background-color: red; }")
                risk_level = "High Risk"

            if prediction == "High Risk":
                self.prediction_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                self.prediction_label.setStyleSheet("color: green; font-weight: bold;")

            self.update_risk_interpretation(risk_level, probability, age_group, diagnosis_category, 
                                          number_inpatient, number_emergency)

            if features_importance is not None:
                self.update_feature_importance_table(features_importance)
            else:
                self.update_feature_importance_with_rules()

            self.update_risk_factors(
                age_group, gender, race, time_in_hospital, num_procedures,
                num_medications, number_emergency, number_inpatient,
                number_diagnoses, diagnosis_category, a1c_result, glu_serum,
                diabetes_med, med_change, insulin, checked_medications
            )
            
            print("Prediction complete")
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Prediction Error", f"Error making prediction: {str(e)}")
            
    def predict_with_model(self, age_group, gender, race, admission_type, admission_source, 
                         time_in_hospital, num_lab_procedures, num_procedures, num_medications,
                         number_outpatient, number_emergency, number_inpatient,
                         number_diagnoses, diagnosis_category, medical_specialty,
                         a1c_result, glu_serum, diabetes_med, med_change, insulin,
                         medications_taken):
        """Make prediction using the trained model"""
        try:

            map_age = {
                '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3,
                '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7,
                '[80-90)': 8, '[90-100)': 9
            }

            input_data = {
                'age': map_age[age_group],
                'gender': gender,
                'race': race,
                'admission_type_id': admission_type,
                'admission_source_id': admission_source,
                'discharge_disposition_id': 'Discharged to home',  
                'time_in_hospital': time_in_hospital,
                'num_lab_procedures': num_lab_procedures,
                'num_procedures': num_procedures,
                'num_medications': num_medications,
                'number_outpatient': number_outpatient,
                'number_emergency': number_emergency,
                'number_inpatient': number_inpatient,
                'number_diagnoses': number_diagnoses,
                'A1Cresult': a1c_result,
                'max_glu_serum': glu_serum,
                'medical_specialty': medical_specialty,
                'diabetesMed': 1 if diabetes_med == 'Yes' else 0,
                'change': med_change,
                'insulin': insulin,
                'payer_code': 'unknown',  
                'diag_category': diagnosis_category
            }

            for category, (lower, upper) in self.diagnosis_categories.items():
                if diagnosis_category == category:

                    diag_code = str((lower + upper) // 2)
                    input_data['diag_1'] = diag_code
                    break
            else:

                input_data['diag_1'] = '250'

            input_data['diag_2'] = 'NO_SECONDARY_DX'
            input_data['diag_3'] = 'NO_TERTIARY_DX'

            for med_name, status in medications_taken.items():
                input_data[med_name] = status

            all_med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                          'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
                          'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
                          'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin',
                          'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
            
            for med in all_med_cols:
                if med not in input_data:
                    input_data[med] = 'No'

            input_data['num_medications_taken'] = sum(1 for med, status in medications_taken.items() if status != 'No')

            input_df = pd.DataFrame([input_data])

            for col in self.column_names:
                if col not in input_df.columns:
                    input_df[col] = 0  
            
            input_df = input_df[self.column_names]  

            categorical_cols = input_df.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                if col in self.encoders:
                    le = self.encoders[col]

                    if input_df[col].iloc[0] in le.classes_:
                        input_df[col] = le.transform(input_df[col])
                    else:

                        print(f"Warning: Value '{input_df[col].iloc[0]}' not found in trained encoder for {col}")
                        input_df[col] = le.transform([le.classes_[0]])

            numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                          'num_medications', 'number_outpatient', 'number_emergency',
                          'number_inpatient', 'number_diagnoses', 'num_medications_taken']

            for col in numeric_cols:
                if col in input_df.columns:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            
            input_df[numeric_cols] = self.scaler.transform(input_df[numeric_cols])

            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(input_df)[0][1]
            else:

                pred = self.model.predict(input_df)[0]
                probability = 0.8 if pred == 1 else 0.2  

            features_importance = []
            
            if hasattr(self, 'feature_importance'):

                top_features = self.feature_importance.head(10)
                features_importance = list(zip(top_features['feature'], top_features['importance']))
            elif hasattr(self.model, 'feature_importances_'):

                importances = self.model.feature_importances_
                features_importance = list(zip(self.column_names, importances))
                features_importance.sort(key=lambda x: x[1], reverse=True)
                features_importance = features_importance[:10]  
            
            return probability, features_importance
            
        except Exception as e:
            print(f"Error in ML prediction: {str(e)}")
            import traceback
            traceback.print_exc()

            return self.predict_with_rules(
                age_group, gender, time_in_hospital, num_procedures,
                num_medications, number_emergency, number_inpatient,
                number_diagnoses, diagnosis_category, a1c_result, glu_serum,
                insulin
            ), None
    
    def predict_with_rules(self, age_group, gender, time_in_hospital, num_procedures,
                          num_medications, number_emergency, number_inpatient,
                          number_diagnoses, diagnosis_category, a1c_result, glu_serum,
                          insulin):
        print(f"Risk calculation for: Age={age_group}, Gender={gender}, Diagnosis={diagnosis_category}")
        print(f"Lab results: A1C={a1c_result}, Glucose={glu_serum}")
        print(f"Previous visits: Inpatient={number_inpatient}, Emergency={number_emergency}")
        
        risk_factors = 0
        
        age_index = self.age_groups.index(age_group)
        if age_index <= 2:
            risk_factors += 0.5
            print(f"Age {age_group}: +0.5 (young adult, low risk)")
        elif age_index <= 4:
            risk_factors += 1.0
            print(f"Age {age_group}: +1.0 (middle-aged adult, moderate risk)")
        elif age_index <= 6:
            risk_factors += 2.0
            print(f"Age {age_group}: +2.0 (older adult, elevated risk)")
        elif age_index <= 7:
            risk_factors += 3.0
            print(f"Age {age_group}: +3.0 (elderly, high risk)")
        else:
            risk_factors += 5.0
            print(f"Age {age_group}: +5.0 (very elderly, very high risk)")
        
        if gender == 'Female':
            risk_factors += 0.2
            print(f"Gender {gender}: +0.2 (slightly higher risk)")
        
        if number_inpatient == 0:
            print("Previous hospitalizations: +0 (none)")
        elif number_inpatient <= 2:
            risk_factors += (number_inpatient * 1.5)
            print(f"Previous hospitalizations ({number_inpatient}): +{number_inpatient * 1.5} (moderate risk)")
        elif number_inpatient <= 5:
            risk_factors += (number_inpatient * 2.0)
            print(f"Previous hospitalizations ({number_inpatient}): +{number_inpatient * 2.0} (high risk)")
        else:
            risk_factors += (5 * 2.0) + ((number_inpatient - 5) * 1.0)
            print(f"Previous hospitalizations ({number_inpatient}): +{(5 * 2.0) + ((number_inpatient - 5) * 1.0)} (very high risk)")
        
        if number_emergency == 0:
            print("Emergency visits: +0 (none)")
        elif number_emergency <= 2:
            risk_factors += (number_emergency * 1.0)
            print(f"Emergency visits ({number_emergency}): +{number_emergency * 1.0} (moderate risk)")
        elif number_emergency <= 5:
            risk_factors += (number_emergency * 1.5)
            print(f"Emergency visits ({number_emergency}): +{number_emergency * 1.5} (high risk)")
        else:
            risk_factors += (5 * 1.5) + ((number_emergency - 5) * 0.8)
            print(f"Emergency visits ({number_emergency}): +{(5 * 1.5) + ((number_emergency - 5) * 0.8)} (very high risk)")
        
        high_risk_diagnoses = ['Heart & Circulatory Conditions', 'Respiratory Diseases', 
                            'Kidney & Urinary Disorders', 'Endocrine Disorders']
        moderate_risk_diagnoses = ['Cancer & Neoplasms', 'Digestive System Diseases', 
                                 'Blood Disorders', 'Mental Health Disorders']
        
        if diagnosis_category in high_risk_diagnoses:
            risk_factors += 3.0
            print(f"Diagnosis {diagnosis_category}: +3.0 (high-risk condition)")
        elif diagnosis_category in moderate_risk_diagnoses:
            risk_factors += 1.5
            print(f"Diagnosis {diagnosis_category}: +1.5 (moderate-risk condition)")
        else:
            risk_factors += 0.5
            print(f"Diagnosis {diagnosis_category}: +0.5 (lower-risk condition)")
        
        if a1c_result == 'not measured':
            risk_factors += 0.5
            print(f"A1C {a1c_result}: +0.5 (not measured - slight risk)")
        elif a1c_result == 'Norm':
            risk_factors += 0.0
            print(f"A1C {a1c_result}: +0.0 (normal - good control)")
        elif a1c_result == '>7':
            risk_factors += 1.5
            print(f"A1C {a1c_result}: +1.5 (poor control)")
        elif a1c_result == '>8':
            risk_factors += 3.0
            print(f"A1C {a1c_result}: +3.0 (very poor control)")
        
        if glu_serum == 'not measured':
            risk_factors += 0.5
            print(f"Glucose {glu_serum}: +0.5 (not measured - slight risk)")
        elif glu_serum == 'Norm':
            risk_factors += 0.0
            print(f"Glucose {glu_serum}: +0.0 (normal - good control)")
        elif glu_serum == '>200':
            risk_factors += 2.0
            print(f"Glucose {glu_serum}: +2.0 (poor control)")
        elif glu_serum == '>300':
            risk_factors += 4.0
            print(f"Glucose {glu_serum}: +4.0 (very poor control)")
        
        if number_diagnoses <= 3:
            print(f"Number of diagnoses ({number_diagnoses}): +0.0 (few diagnoses)")
        elif number_diagnoses <= 6:
            risk_factors += 1.0
            print(f"Number of diagnoses ({number_diagnoses}): +1.0 (moderate complexity)")
        elif number_diagnoses <= 9:
            risk_factors += 2.0
            print(f"Number of diagnoses ({number_diagnoses}): +2.0 (high complexity)")
        else:
            risk_factors += 3.0
            print(f"Number of diagnoses ({number_diagnoses}): +3.0 (very high complexity)")
        
        if num_procedures <= 1:
            print(f"Number of procedures ({num_procedures}): +0.0 (few procedures)")
        elif num_procedures <= 3:
            risk_factors += 0.5
            print(f"Number of procedures ({num_procedures}): +0.5 (moderate procedures)")
        elif num_procedures <= 6:
            risk_factors += 1.5
            print(f"Number of procedures ({num_procedures}): +1.5 (many procedures)")
        else:
            risk_factors += 2.5
            print(f"Number of procedures ({num_procedures}): +2.5 (very many procedures)")
        
        if num_medications <= 5:
            print(f"Medications count ({num_medications}): +0.0 (few medications)")
        elif num_medications <= 10:
            risk_factors += 0.5
            print(f"Medications count ({num_medications}): +0.5 (moderate medications)")
        elif num_medications <= 20:
            risk_factors += 1.0
            print(f"Medications count ({num_medications}): +1.0 (many medications)")
        else:
            risk_factors += 2.0
            print(f"Medications count ({num_medications}): +2.0 (polypharmacy)")
        
        if time_in_hospital <= 3:
            print(f"Hospital stay ({time_in_hospital} days): +0.0 (short stay)")
        elif time_in_hospital <= 7:
            risk_factors += 0.5
            print(f"Hospital stay ({time_in_hospital} days): +0.5 (moderate stay)")
        elif time_in_hospital <= 14:
            risk_factors += 1.5
            print(f"Hospital stay ({time_in_hospital} days): +1.5 (long stay)")
        else:
            risk_factors += 3.0
            print(f"Hospital stay ({time_in_hospital} days): +3.0 (very long stay)")
        
        if insulin == 'No':
            print(f"Insulin {insulin}: +0.0 (not used)")
        elif insulin == 'Steady':
            risk_factors += 1.0
            print(f"Insulin {insulin}: +1.0 (stable regimen)")
        elif insulin == 'Down':
            risk_factors += 1.5
            print(f"Insulin {insulin}: +1.5 (changing regimen - down)")
        elif insulin == 'Up':
            risk_factors += 2.5
            print(f"Insulin {insulin}: +2.5 (changing regimen - up)")
        
        max_risk_factors = 35.0
        raw_probability = risk_factors / max_risk_factors
        
        if age_index <= 2 and number_inpatient == 0 and number_emergency == 0 and diagnosis_category not in high_risk_diagnoses:
            probability = min(0.4, raw_probability)
            print(f"Applied young/healthy cap: {probability:.2f}")
        elif age_index >= 8 and (number_inpatient > 3 or number_emergency > 3) and diagnosis_category in high_risk_diagnoses:
            probability = max(0.75, raw_probability)
            print(f"Applied elderly/high-risk minimum: {probability:.2f}")
        else:
            probability = min(0.95, raw_probability)
            
        print(f"Total risk factors: {risk_factors:.1f}/{max_risk_factors:.1f}")
        print(f"Final probability: {probability:.2f}")
        
        return probability
    
    def update_risk_interpretation(self, risk_level, probability, age_group, diagnosis_category, 
                                  number_inpatient, number_emergency):
        interpretations = {
            "Low Risk": "Based on the provided information, this patient has a low risk of hospital readmission within 30 days. Continue with standard care protocols.",
            "Moderate Risk": "This patient has a moderate risk of hospital readmission. Consider additional follow-up care and patient education to reduce this risk.",
            "High Risk": "This patient has a high risk of hospital readmission within 30 days. Recommend developing a comprehensive discharge plan with close follow-up care."
        }
        
        interpretation = interpretations[risk_level]
        
        specifics = []
        
        if age_group:
            age_index = self.age_groups.index(age_group)
            if age_index >= 8:
                specifics.append(f"The patient's advanced age ({age_group}) is a significant risk factor for readmission.")
            elif age_index >= 6:
                specifics.append(f"The patient's age ({age_group}) is a contributing risk factor.")
        
        high_risk_diagnoses = ['Heart & Circulatory Conditions', 'Respiratory Diseases', 
                            'Kidney & Urinary Disorders', 'Endocrine Disorders']
        if diagnosis_category in high_risk_diagnoses:
            specifics.append(f"The primary diagnosis of {diagnosis_category} substantially increases readmission risk.")
        
        if number_inpatient > 0:
            if number_inpatient >= 10:
                specifics.append(f"Previous inpatient visits ({number_inpatient}) indicate a very high risk of readmission based on extensive hospital utilization history.")
            elif number_inpatient >= 5:
                specifics.append(f"Previous inpatient visits ({number_inpatient}) indicate a high risk of readmission based on hospital utilization history.")
            else:
                specifics.append(f"Previous inpatient visits ({number_inpatient}) indicate a history of hospital utilization.")
        
        if number_emergency > 0:
            if number_emergency >= 5:
                specifics.append(f"Multiple emergency visits ({number_emergency}) strongly suggest inadequate disease management and care coordination.")
            else:
                specifics.append(f"Previous emergency visits ({number_emergency}) suggest potential care management issues.")
        
        if probability >= 0.7:
            specifics.append("This patient should receive intensive follow-up care and care coordination services to prevent readmission.")
        elif probability >= 0.4:
            specifics.append("Consider scheduling follow-up appointments within 7 days of discharge and implementing medication reconciliation.")
        
        if specifics:
            interpretation += "\n\n" + " ".join(specifics)
        
        self.risk_interpretation.setText(interpretation)
    
    def update_feature_importance_table(self, features_importance):
        self.feature_table.clearContents()
        
        self.feature_table.setRowCount(len(features_importance))
        
        for i, (feature, importance) in enumerate(features_importance):
            readable_feature = feature.replace('_', ' ').title()
            self.feature_table.setItem(i, 0, QTableWidgetItem(readable_feature))
            
            importance_item = QTableWidgetItem(f"{importance:.4f}")
            self.feature_table.setItem(i, 1, importance_item)
            
            if importance > 0.1:
                importance_item.setBackground(QColor('#ffcccc'))
            elif importance > 0.05:
                importance_item.setBackground(QColor('#ffffcc'))
            else:
                importance_item.setBackground(QColor('#e6ffe6'))
    
    def update_feature_importance_with_rules(self):
        features = [
            ('Previous Inpatient Visits', 0.20),
            ('Previous Emergency Visits', 0.18),
            ('Primary Diagnosis', 0.15),
            ('Age', 0.12),
            ('Glucose Serum Test', 0.10),
            ('A1C Result', 0.08),
            ('Insulin', 0.06),
            ('Number of Diagnoses', 0.05),
            ('Time in Hospital', 0.04),
            ('Number of Medications', 0.02)
        ]
        
        self.update_feature_importance_table(features)
    
    def update_risk_factors(self, age_group, gender, race, time_in_hospital, num_procedures,
                          num_medications, number_emergency, number_inpatient,
                          number_diagnoses, diagnosis_category, a1c_result, glu_serum,
                          diabetes_med, med_change, insulin, checked_medications):
        """Update the risk factors analysis"""

        risk_text = f"Patient Profile: {gender}, {age_group} age range, {race}.\n\n"
        risk_text += "This analysis identifies key factors that may contribute to readmission risk. "
        risk_text += "These should be addressed in the discharge and follow-up care plan."
        
        self.risk_factors_text.setText(risk_text)

        self.risk_factors_table.clearContents()

        risk_factors = []

        age_index = self.age_groups.index(age_group)
        if age_index >= 6:  
            if age_index >= 8:  
                risk_factors.append(("Age", f"{age_group} (Elevated Risk)", QColor('#ffcccc')))
            else:
                risk_factors.append(("Age", f"{age_group} (Moderate Risk)", QColor('#ffffcc')))
        else:
            risk_factors.append(("Age", f"{age_group} (Low Risk)", QColor('#e6ffe6')))

        high_risk_diagnoses = ['Heart & Circulatory Conditions', 'Respiratory Diseases', 
                             'Kidney & Urinary Disorders', 'Endocrine Disorders']
        if diagnosis_category in high_risk_diagnoses:
            risk_factors.append(("Diagnosis", f"{diagnosis_category} (High Risk)", QColor('#ffcccc')))
        else:
            risk_factors.append(("Diagnosis", f"{diagnosis_category} (Lower Risk)", QColor('#e6ffe6')))

        if number_inpatient > 0:
            if number_inpatient >= 10:
                risk_factors.append(("Previous Hospitalizations", f"{number_inpatient} (Very High Risk)", QColor('#ff9999')))
            elif number_inpatient >= 5:
                risk_factors.append(("Previous Hospitalizations", f"{number_inpatient} (High Risk)", QColor('#ffcccc')))
            else:
                risk_factors.append(("Previous Hospitalizations", f"{number_inpatient} (Elevated Risk)", QColor('#ffffcc')))
        else:
            risk_factors.append(("Previous Hospitalizations", "None (Low Risk)", QColor('#e6ffe6')))

        if number_emergency > 0:
            if number_emergency >= 10:
                risk_factors.append(("Emergency Visits", f"{number_emergency} (Very High Risk)", QColor('#ff9999')))
            elif number_emergency >= 5:
                risk_factors.append(("Emergency Visits", f"{number_emergency} (High Risk)", QColor('#ffcccc')))
            else:
                risk_factors.append(("Emergency Visits", f"{number_emergency} (Elevated Risk)", QColor('#ffffcc')))
        else:
            risk_factors.append(("Emergency Visits", "None (Low Risk)", QColor('#e6ffe6')))

        if a1c_result in ['>7', '>8']:
            risk_factors.append(("A1C Level", f"{a1c_result} (Poor Control)", QColor('#ffcccc')))
        else:
            risk_factors.append(("A1C Level", f"{a1c_result} (Better Control)", QColor('#e6ffe6')))

        if glu_serum in ['>200', '>300']:
            risk_factors.append(("Glucose Level", f"{glu_serum} (Poor Control)", QColor('#ffcccc')))
        else:
            risk_factors.append(("Glucose Level", f"{glu_serum} (Better Control)", QColor('#e6ffe6')))

        if time_in_hospital > 7:
            if time_in_hospital >= 20:
                risk_factors.append(("Hospital Stay", f"{time_in_hospital} days (Very Long)", QColor('#ffcccc')))
            else:
                risk_factors.append(("Hospital Stay", f"{time_in_hospital} days (Extended)", QColor('#ffffcc')))
        else:
            risk_factors.append(("Hospital Stay", f"{time_in_hospital} days (Normal)", QColor('#e6ffe6')))

        if number_diagnoses > 5:
            if number_diagnoses >= 10:
                risk_factors.append(("Multiple Diagnoses", f"{number_diagnoses} (Very Complex)", QColor('#ffcccc')))
            else:
                risk_factors.append(("Multiple Diagnoses", f"{number_diagnoses} (Complex Case)", QColor('#ffffcc')))
        else:
            risk_factors.append(("Multiple Diagnoses", f"{number_diagnoses} (Less Complex)", QColor('#e6ffe6')))

        if diabetes_med == 'Yes':
            risk_factors.append(("Diabetes Medication", "Yes (Indicates Disease)", QColor('#ffffcc')))
        else:
            risk_factors.append(("Diabetes Medication", "No", QColor('#e6ffe6')))

        if med_change == 'Ch':
            risk_factors.append(("Medication Change", "Recent Change (Unstable)", QColor('#ffcccc')))
        else:
            risk_factors.append(("Medication Change", "No Change (Stable)", QColor('#e6ffe6')))

        if insulin != 'No':
            if insulin == 'Up':
                risk_factors.append(("Insulin", "Increased (High Risk)", QColor('#ffcccc')))
            elif insulin == 'Down':
                risk_factors.append(("Insulin", "Decreased (Moderate Risk)", QColor('#ffffcc')))
            else:
                risk_factors.append(("Insulin", "Steady (Controlled)", QColor('#e6ffe6')))
        else:
            risk_factors.append(("Insulin", "Not Used", QColor('#e6ffe6')))

        if num_medications > 20:
            risk_factors.append(("Number of Medications", f"{num_medications} (Polypharmacy)", QColor('#ffcccc')))
        else:
            risk_factors.append(("Number of Medications", f"{num_medications}", QColor('#e6ffe6')))

        if num_procedures > 5:
            risk_factors.append(("Number of Procedures", f"{num_procedures} (Complex Case)", QColor('#ffcccc')))
        else:
            risk_factors.append(("Number of Procedures", f"{num_procedures}", QColor('#e6ffe6')))

        if checked_medications > 0:
            risk_factors.append(("Specific Medications", f"{checked_medications} medications selected", QColor('#ffffcc')))

        if self.number_outpatient.value() > 0:
            risk_factors.append(("Outpatient Visits", f"{self.number_outpatient.value()}", QColor('#e6ffe6')))

        self.risk_factors_table.setRowCount(len(risk_factors))

        for i, (factor, status, color) in enumerate(risk_factors):
            self.risk_factors_table.setItem(i, 0, QTableWidgetItem(factor))
            status_item = QTableWidgetItem(status)
            status_item.setBackground(color)
            self.risk_factors_table.setItem(i, 1, status_item)

    def display_results(self, probability, risk_factors):
        """Display prediction results and risk factors"""

        if probability < 0.3:
            risk_level = "Low Risk"
            color = "green"
        elif probability < 0.6:
            risk_level = "Moderate Risk"
            color = "orange"
        else:
            risk_level = "High Risk"
            color = "red"

        prediction_text = f"Readmission Prediction: {risk_level} ({probability:.1%})"
        self.prediction_label.setText(prediction_text)
        self.prediction_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14pt;")

        self.risk_progress.setValue(int(probability * 100))
        self.risk_progress.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")

        self.risk_factors_table.clearContents()
        self.risk_factors_table.setRowCount(min(len(risk_factors), 5))
        
        for i, factor in enumerate(risk_factors[:5]):
            self.risk_factors_table.setItem(i, 0, QTableWidgetItem("Risk Factor"))
            status_item = QTableWidgetItem(factor)
            status_item.setBackground(QColor(color))
            self.risk_factors_table.setItem(i, 1, status_item)

        self.update_risk_interpretation(risk_level, probability, "", "", 0, 0)

if __name__ == "__main__":
    try:
        print("Starting QApplication...")
        app = QApplication(sys.argv)
        print("Creating main window...")
        window = HospitalReadmissionGUI()
        print("Showing window...")
        window.show()
        print("Entering event loop...")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc() 