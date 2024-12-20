
#imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from functions import *
import streamlit as st

with st.sidebar:
    st.write('this is a sidebar')

st.title('Titanic Project')

st.header('Data Exploration')
#downloading the data
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv'
titanic_df = get_data(url)

#backup
titanic_df_backup = titanic_df.copy()
st.subheader('Dataset')
st.write(titanic_df.head(5))

#feature engineering
titanic_df.Sex.replace({'male':0, 'female':1}, inplace=True)

#plots
titanic_corr = titanic_df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(titanic_corr, annot=True, ax=ax)


st.subheader('Feature comparisons')
col_1, col_2 = st.columns(2)

with col_1:
    fig, axes = plt.subplots(figsize=(8, 4))
    ax = axes
    #titanic_df.Age.hist(ax=ax, bins=30)
    ax.hist(titanic_df.Age, bins=40)
    ax.set_xlabel('Age ranges')
    ax.set_ylabel('Quantity')
    st.pyplot(fig)
    st.caption('Distribution of ages')

with col_2:
    fig, axes = plt.subplots(figsize=(8, 4))
    ax = axes
    titanic_df.Fare.hist(ax=ax, bins=40)
    ax.set_xlabel('Fare ranges')
    ax.set_ylabel('Quantity')
    st.write(fig)
    st.caption('Distribution of fares')

st.write('The distribution of ages shows a normal distribution, while the fares are highly skewed to the left')
#survival prediction
titanic_df[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean()

with st.expander('Model'):
    select_model = st.selectbox('Select a model: ', ['Random Forest', 'Gaussian NB'])
    model = GaussianNB()
    if select_model == 'Random Forest':
        model = RandomForestClassifier()

    features = ['Sex', 'Pclass', 'Fare']
    select_features = st.multiselect('Features:', features)
    if len(select_features) != 0 and st.button('RUN MODEL'):
        x = titanic_df[select_features]
        y = titanic_df.Survived
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        st.write('Accuracy: ' + str(accuracy_score(y_pred, y_test)))

