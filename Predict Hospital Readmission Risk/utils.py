import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import train_and_save_model; train_and_save_model()
import joblib

def train_and_save_model():
    df = pd.read_csv('data/healthcare_dataset.csv')

    df = df.dropna(subset=['Test Results'])
    df['Room Number'] = df['Room Number'].fillna('Unknown')
    df['Admission Type'] = df['Admission Type'].fillna('Unknown')
    df['Discharge Date'] = df['Discharge Date'].fillna('Unknown')
    df['Medication'] = df['Medication'].fillna('Unknown')

    X = df.drop('Test Results', axis=1)
    y = df['Test Results']

    cat_cols = ['Gender', 'Blood Type', 'Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication']
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.select_dtypes(include=['number'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'model/readmission_model.pkl')
    return model, X.columns.tolist()
