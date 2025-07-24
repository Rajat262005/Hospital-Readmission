import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model and features
model = joblib.load('model/readmission_model.pkl')
df = pd.read_csv('data/healthcare_dataset.csv')
feature_cols = ['Age', 'Room Number', 'Gender', 'Blood Type', 'Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication']

st.set_page_config(page_title="🏥 Hospital Readmission Dashboard", layout="wide")

# Sidebar
st.sidebar.title("🔍 Prediction Form")
gender = st.sidebar.selectbox("Gender", df['Gender'].dropna().unique())
age = st.sidebar.slider("Age", 0, 100, 50)
med_cond = st.sidebar.selectbox("Medical Condition", df['Medical Condition'].dropna().unique())
admission_type = st.sidebar.selectbox("Admission Type", df['Admission Type'].dropna().unique())
medication = st.sidebar.selectbox("Medication", df['Medication'].dropna().unique())
room_number = st.sidebar.number_input("Room Number", value=101)
blood_type = st.sidebar.selectbox("Blood Type", df['Blood Type'].dropna().unique())
insurance = st.sidebar.selectbox("Insurance Provider", df['Insurance Provider'].dropna().unique())

input_df = pd.DataFrame({
    'Age': [age],
    'Room Number': [room_number],
    'Gender': [gender],
    'Blood Type': [blood_type],
    'Medical Condition': [med_cond],
    'Insurance Provider': [insurance],
    'Admission Type': [admission_type],
    'Medication': [medication]
})

# Encoding
from sklearn.preprocessing import LabelEncoder
cat_cols = ['Gender', 'Blood Type', 'Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication']
le = LabelEncoder()
for col in cat_cols:
    input_df[col] = le.fit_transform(input_df[col].astype(str))

# Select numeric only
input_df = input_df.select_dtypes(include=['number'])

# Prediction
if st.sidebar.button("Predict Readmission"):
    prediction = model.predict(input_df)[0]
    st.sidebar.success(f"🔁 Readmission Risk Prediction: **{prediction}**")

# Dashboard
st.title("📊 Hospital Readmission Dashboard")

tabs = st.tabs(["📋 Dataset", "📈 Visualizations", "🌟 Features", "📊 Confusion Matrix"])

with tabs[0]:
    st.subheader("Sample of Dataset")
    st.dataframe(df.head())

with tabs[1]:
    st.subheader("Readmission by Gender")
    fig = px.pie(df, names='Gender', title='Readmission Count by Gender')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Readmission by Medical Condition")
    fig2 = px.pie(df, names='Medical Condition', title='Readmission Count by Condition')
    st.plotly_chart(fig2, use_container_width=True)

with tabs[2]:
    st.subheader("Top Features Importance")
    importances = model.feature_importances_
    features_df = pd.DataFrame({'Feature': input_df.columns, 'Importance': importances})
    features_df = features_df.sort_values(by='Importance', ascending=False)
    st.bar_chart(features_df.set_index("Feature"))

with tabs[3]:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    X = df.drop('Test Results', axis=1)
    y = df['Test Results']
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))
    X = X.select_dtypes(include=['number'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix")
    st.write(cm)
    st.write("Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.write("Accuracy Score:", accuracy_score(y_test, y_pred))