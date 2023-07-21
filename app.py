import streamlit as st
import pickle
import numpy as np
def load_model():
    with open ('saved_steps1.pkl','rb') as file:
        data=pickle.load(file)  
    return data
dataset=load_model()
classifier=dataset["classifier"]
l_race=dataset["l_race"]
l_gender=dataset["l_gender"]
l_specialty=dataset["l_specialty"]
l_examide=dataset["l_examide"]
l_insulin=dataset["l_insulin"]
l_glipizide_metformin=dataset["l_glipizide_metformin"]
l_diabetesMed=dataset["l_diabetesMed"]


st.title("WELCOME TO DIABETES PREDICTION SYSTEM")
st.write("""### We need some information !""")
races=(
    "Caucasian","AfricanAmerican","Hispanic","Asian","Other"
)
genders=(
    "Male","Female"
)
specialties=(
    "Medical","Non medical","Psychiatry"
)
glu=(
    "Norm","Abnorm"
)
races=(
    "Caucasian","AfricanAmerican","Hispanic","Asian","Other"
)
examides=(
    "No","Yes"
)
insulins=(
    "No","Yes"
)
glipizide_metformin=(
    "Steady","No"
)

diabetesMeds=(
    "No","Yes"
)
race=st.selectbox("race",races)
gender=st.selectbox("gender",genders)
age=st.slider("age",0,90,10)
time_in_hospital=st.slider("time_in_hospital",0,15,7)
specialty=st.selectbox("specialty",specialties)
num_lab_procedures=st.slider("num_lab_procedures",0,50,10)
num_procedures=st.slider("num_procedures",0,7,3)
num_medications=st.slider("num_medications",0,15,10)
number_outpatient=st.slider("number_outpatient",0,10,5)
number_emergency=st.slider("number_emergency",0,15,5)
number_inpatient=st.slider("number_inpatient",0,12,6)
number_diagnoses=st.slider("number_diagnoses",0,10,5)
examide=st.selectbox("examide",examides)
insulin=st.selectbox("insulin",insulins)
glipizide_metformin=st.selectbox("glipizide_metformin",glipizide_metformin)
diabetesMed=st.selectbox("diabetesMed",diabetesMeds)
ok=st.button("Submit")
if ok:
    X=np.array([[race, gender, age, time_in_hospital, specialty,
    num_lab_procedures, num_procedures, num_medications,
    number_outpatient, number_emergency, number_inpatient,
    number_diagnoses, examide, insulin, glipizide_metformin,
    diabetesMed]])
    X[:,0]=l_race.transform(X[:,0])
    X[:,1]=l_gender.transform(X[:,1])
    X[:,4]=l_specialty.transform(X[:,4])
    X[:,12]=l_examide.transform(X[:,12])
    X[:,13]=l_insulin.transform(X[:,13])
    X[:,14]=l_glipizide_metformin.transform(X[:,14])
    X[:,15]=l_diabetesMed.transform(X[:,15])
    X=X.astype(float)
    prediction=classifier.predict(X)
    if prediction==1:
        st.subheader("You are diabetic")
    else :
        st.subheader("You are not diabetic")



        
# 'race', 'gender', 'age', 'time_in_hospital', 'specialty',
#        'num_lab_procedures', 'num_procedures', 'num_medications',
#        'number_outpatient', 'number_emergency', 'number_inpatient',
#        'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin',
#        'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
#        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
#        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
#        'tolazamide', 'examide', 'citoglipton', 'insulin',
#        'glyburide.metformin', 'glipizide.metformin',
#        'glimepiride.pioglitazone', 'metformin.rosiglitazone',
#        'metformin.pioglitazone', 'diabetesMed', 'readmitted'