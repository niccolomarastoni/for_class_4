
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""**EX**: Write a function that takes an URL/path and returns a DataFrame if that URL/path is a valid csv file"""

def get_data(url : str) -> pd.DataFrame:
    '''
    get_data: given a url, outputs a dataframe with the right data
    attributes:
    url: string

    examples:
    >> url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv'
    >> titanic_df = get_data(url)
    '''
    try:
        data_df = pd.read_csv(url)
        return data_df
    except:
        print('Not a proper csv file')

a_string = ''' hello
this
works
'''

print(a_string)

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv'
titanic_df = get_data(url)

titanic_df.head(2)

print(get_data.__doc__)

print(pd.concat.__doc__)

titanic_df.info()

titanic_df.describe().T

titanic_df.Sex.unique()

titanic_df_backup = titanic_df.copy()

titanic_df.Sex.replace({'male':0, 'female':1}, inplace=True)

titanic_df.Sex[:2]

import seaborn as sns

titanic_corr = titanic_df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(titanic_corr, annot=True, ax=ax)

"""Comments about the correlation matrix

**Ex**: Plot the distribution of Age and Fare using histograms
"""

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

"""comments about the histogram

**EX**: what is the percentage of survived per Pclass?
"""

titanic_df.columns

titanic_df[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean()

"""#Predict who survives"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

titanic_df.columns

features = ['Sex', 'Pclass', 'Fare']
x = titanic_df[features]
y = titanic_df.Survived
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_pred, y_test))

