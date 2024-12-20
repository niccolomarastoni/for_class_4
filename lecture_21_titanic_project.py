
#imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from functions import *
import streamlit as st

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



fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes[0]
#titanic_df.Age.hist(ax=ax, bins=30)
ax.hist(titanic_df.Age, bins=40)
ax.set_xlabel('Age ranges')
ax.set_ylabel('Quantity')
ax = axes[1]
titanic_df.Fare.hist(ax=ax, bins=40)
ax.set_xlabel('Fare ranges')
ax.set_ylabel('Quantity')
plt.show()

#survival prediction
titanic_df[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean()

features = ['Sex', 'Pclass', 'Fare']
x = titanic_df[features]
y = titanic_df.Survived
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_pred, y_test))

