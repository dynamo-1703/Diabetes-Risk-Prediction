import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import streamlit as st
from sklearn.metrics import confusion_matrix

# Load the Data
@st.cache_data
def load_data():
    return pd.read_csv('diabetes_sataset.csv')

df = load_data()

print(df.shape) 
print(df.isnull().sum())  #no null value
print(df.describe())
print(df.info())

# Assigning Varibles
X = df.drop(["Outcome"] , axis =1)
y = df['Outcome']

# spliting data
X_train,X_test ,y_train ,y_test = train_test_split(X,y , test_size = 0.2 , random_state = 42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

# Fit the model
grid_search.fit(X_train, y_train)

# Best estimator
best_rf = grid_search.best_estimator_

# saving the best mode
joblib.dump(best_rf, "rf_model.pkl")



#  Streamlit part


st.title("🩺 Diabetes Risk Prediction App")
st.divider()

# NAVBAR
st.sidebar.title("NAVIGATION")
page = st.sidebar.radio("Navigation", ['HOME', 'DATA VISUALIZATION', 'DIABETES PREDICTION'])

if page == "HOME":
    st.title("About the App ")
    st.markdown("""
       This interactive web app uses a machine learning model trained on health data to **predict whether a person is likely to have diabetes**.

       The model behind the scenes is a **Random Forest Classifier**, trained on a medical dataset.

       Enter patient health indicators to get a quick risk assessment.
    """)

    st.markdown("---")
    st.markdown("Prediction Accuracy  =  74%")
    st.markdown('**Confusion matrix :**')
    #  matrix

    # Load model
    model = joblib.load("rf_model.pkl")

    # predict
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test , y_pred)
    labels = ["Non-Diabetic", "Diabetic"]

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",xticklabels=labels, yticklabels=labels)
            
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    st.pyplot(fig)


elif page == "DATA VISUALIZATION":
    st.title("📊 Data Visualization")
    st.markdown("""
    The data used to train this model comes from the **Pima Indians Diabetes Database**, a well-known medical dataset.

    It includes **768 female patients** of Pima Indian heritage, aged **21 years or older**. Each patient is characterized by **8 medical measurements**, including:


    - **Pregnancies**: Number of times pregnant  
    - **Glucose**: Plasma glucose concentration  
    - **Blood Pressure**: Diastolic blood pressure (mm Hg)  
    - **Skin Thickness**: Triceps skinfold thickness (mm)  
    - **Insulin**: 2-hour serum insulin (mu U/ml)  
    - **BMI**: Body mass index (weight/height²)  
    - **Diabetes Pedigree Function**: Genetic risk score  
    - **Age**: Age in years  
                
    The target variable is `Outcome`:  
    - `1` = Diabetic  
    - `0` = Not Diabetic
    """)

    st.markdown("### 🔎 Select Feature to Explore Distribution")
    selected_feature = st.selectbox("Select a Feature", df.columns[:-1])  # excluding Outcome
    fig, ax = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, bins=30, ax=ax)
    st.pyplot(fig)


elif page == "DIABETES PREDICTION":
    st.title("🧪 Predict Diabetes Risk")
    st.markdown("Fill in the patient's health details:")
    Pregnancies =st.number_input('Pregnancies' , min_value = 0 , max_value = 20 , value=1)
    Glucose = st.slider("Glucose Level", 0, 200, 120)
    Blood_Pressure= st.slider("Blood Pressurel", 0, 140, 70)
    Skin_Thickness= st.slider("Skin Thickness", 0, 100, 20) 
    Insulin= st.slider("Insulin", 0, 900, 80)
    BMI= st.slider("BMI", 0.0, 70.0, 25.0)
    Diabetes_Pedigree = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    Age= st.slider("Age", 10, 100, 30)

    input_data = np.array([[Pregnancies, Glucose, Blood_Pressure, Skin_Thickness,
                            Insulin, BMI, Diabetes_Pedigree, Age]])
    

    model = joblib.load("rf_model.pkl")

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"🛑 The model predicts that this person is **likely diabetic**.")
        else:
            st.success(f"✅ The model predicts that this person is **not diabetic**.")

        st.markdown(f"### 🔬 Prediction Probability: `{probability:.2f}`")
