import pandas as pd
import streamlit as st
import joblib

st.title("Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model information"])

# === Tab1 ===
with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ['Male', 'Female'])
    chest_pain = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
    resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0, max_value=300)
    cholesterol = st.number_input('Serum Cholesterol (mm/dl)', min_value=0)
    fasting_bs = st.selectbox('Fasting Blood Sugar', ['<= 120 mg/dl', '> 120 mg/dl'])
    resting_ecg = st.selectbox('Resting ECG results', ['Normal', 'ST', 'LVH'])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox('Exercised-Induced Angina', ['Yes', 'No'])
    oldpeak = st.number_input('Oldpeak (ST Depression)', min_value=0.0, max_value=10.0)
    st_slope = st.selectbox('Slope of Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])

    # === Converting Categorical Inputs into Numbers ===
    sex = 0 if sex == 'Male' else 1
    chest_pain = ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'].index(chest_pain)
    fasting_bs = 1 if fasting_bs == '> 120 mg/dl' else 0
    resting_ecg = ['Normal', 'ST', 'LVH'].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == 'Yes' else 0
    st_slope = ['Upsloping', 'Flat', 'Downsloping'].index(st_slope)

    # === Create a Dataframe with User Inputs ===
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    algonames = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'XGBoost']
    modelnames = ['dt.pkl', 'lr.pkl', 'rf.pkl', 'svm.pkl', 'xgb.pkl']


    # === Make Predictions ===
    
    def predict_heart_disease(data):
        predictions = []
        for modelname in modelnames:
            model = joblib.load(modelname)
            prediction = model.predict(data)
            predictions.append(prediction)
        
        return predictions
    
    if st.button('Submit'):
        st.subheader('Results...')
        st.markdown('===========================')

        result = predict_heart_disease(input_data)

        for i in range(len(result)):
            st.subheader(algonames[i])
            if result[i][0] == 0:
                st.write('No Heart Disease Detected')
            else:
                st.write('Heart Disease Detected')
            
            st.markdown('===========================')

# === Tab2 ===
with tab2:
    st.title("Upload CSV File")

    st.subheader("Instructions to note before uploading the file:")
    st.info(
        """
1. No NaN values allowed.
2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP',
   'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope').
3. Check the spellings of the feature names.
4. Feature values conventions:
   - Age: age of the patient [years]
   - Sex: sex of the patient [0: Male, 1: Female]
   - ChestPainType: chest pain type [0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic]
   - RestingBP: resting blood pressure [mm Hg]
   - Cholesterol: serum cholesterol [mg/dl]
   - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
   - RestingECG: resting electrocardiogram results [0: Normal, 1: ST, 2: LVH]
   - MaxHR: maximum heart rate achieved [numeric value between 60 and 202]
   - ExerciseAngina: exercise-induced angina [1: Yes, 0: No]
   - Oldpeak: ST depression induced by exercise relative to rest [numeric]
   - ST_Slope: the slope of the peak exercise ST segment [0: upsloping, 1: flat, 2: downsloping]
        """
    )

    uploaded_file = st.file_uploader('Upload a csv file', type=['csv'])

    if uploaded_file is not None:

        input_data = pd.read_csv(uploaded_file)
        X = input_data.iloc[:, :-1]

        # === Converting Categorical Inputs into Numbers ===
        def encode(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            if 'Sex' in df.columns:
                df['Sex'] = df['Sex'].replace({'M':0, 'Male':0, 'F':1, 'Female':1}).astype('int64')
            if 'ChestPainType' in df.columns:
                df['ChestPainType'] = df['ChestPainType'].replace({
                    'Typical Angina':0, 'TA':0,
                    'Atypical Angina':1, 'ATA':1,
                    'Non-Anginal Pain':2, 'NAP':2,
                    'Asymptomatic':3, 'ASY':3
                })
            if 'FastingBS' in df.columns:
                df['FastingBS'] = df['FastingBS'].replace({'> 120 mg/dl':1, '<= 120 mg/dl':0, '1':1, '0':0}).astype('int64')
            if 'RestingECG' in df.columns:
                df['RestingECG'] = df['RestingECG'].replace({'Normal':0, 'ST':1, 'LVH':2})
            if 'ExerciseAngina' in df.columns:
                df['ExerciseAngina'] = df['ExerciseAngina'].replace({'Yes':1, 'Y':1, 'No':0, 'N':0}).astype('int64')
            if 'ST_Slope' in df.columns:
                df['ST_Slope'] = df['ST_Slope'].replace({'Upsloping':0, 'Up':0, 'Flat':1, 'Downsloping':2, 'Down':2})
            for c in ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            return df

        X = encode(X)

        results = predict_heart_disease(X)
        
        for i in range(len(results)):
            input_data[f'{algonames[i]}'] = results[i]

        st.subheader('Predictions: ')
        st.write(input_data)

        st.download_button(
            "Download predictions CSV",
            data=input_data.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
    else:
        st.info('Upload a CSV file to get predictions.')

# === Tab3 ===
with tab3:
    import plotly.express as px
    data = {
        'Decision Tree': 81,
        'Logistic Regression': 86,
        'Random Forest': 85.3,
        'Support Vector Machine': 83.7,
        'XBoost': 88
    }

    models = list(data.keys())
    accuracies = list(data.values())
    df = pd.DataFrame(list(zip(models, accuracies)), columns=['Models', 'Accuracies %'])
    fig = px.bar(df, y='Accuracies %', x = 'Models')
    st.plotly_chart(fig)
