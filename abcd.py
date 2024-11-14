# pip install -r requirements.txt
import sklearn
import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv('C:\\Users\\v-sarvesh.y\\Downloads\\loan_approval_dataset.csv')
print(df.head())

#from sklearn.datasets import loan_approval_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsRegressor
import time

st.title("LOAN APPROVAL PROCESS")

name = st.text_input("Enter your name")

#if st.button('click me'):
 #   st.write('button clicked:')

 #import numpy as np
 
 #x=np.linespace(0,10,100)
 #y=np.sin(x)
 #st.line_chart(y)

# if st.button('click me'):
# st.markdown('<h1 style='color: red:>hello</h1>', unsafe_allow_html=True')



from PIL import Image
image = Image.open('C:\\Users\\v-sarvesh.y\\Downloads\\loan.png')
st.image(image, caption='LOAN', use_column_width=True)

user_input_income = st.sidebar.slider('enter the income',1,10,3)
user_input_year = st.sidebar.slider('enter the years', 1,5,1)
user_input_loanamount = st.sidebar.slider('enter the loan amount',10000,500000,50000)
user_input_cerditscore = st.sidebar.slider('enter credit score',0,10,1)

X = df.drop(columns=['income_annum','self_employed'],errors='ignore')
Y = df[' income_annum']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#to scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#to creatre a model
regressor = KNeighborsRegressor(n_neighbors=user_input_income)
regressor.fit(X_train_scaled, y_train)

#to create dataframe
user_input_df = pd.DataFrame({
    'Income' : (user_input_income),
    'LoanYear' : (user_input_year),
    'Amount' : (user_input_loanamount),
    'creditscore' : (user_input_cerditscore)
}, columns=X.cpolumns)

#scaler the user input
user_input_scaled = scaler.transform(user_input_df)

#diaplay a loading spinner
with st.spinner('calculating..'):
    time.sleep(3)

#tp predict the loan value
    predicted_med_loan_amount = regressor.predict(user_input_scaled)


# to print the median loan amount
st.write(f' the predicted value of loan amount is:{predicted_med_loan_amount}')













