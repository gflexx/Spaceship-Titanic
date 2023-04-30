#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('titanic_dataset/train.csv')
test = pd.read_csv('titanic_dataset/test.csv')


# In[3]:


train.head()


# In[4]:


train.shape


# In[5]:


test


# In[6]:


train.describe()


# In[7]:


train.dtypes


# In[8]:


train.isna().sum()


# In[ ]:





# In[9]:


test.isna().sum()


# In[4]:


def fill_na_records(df):
    df['HomePlanet'].fillna('UnknownHomePlanet', inplace = True)
    df['Destination'].fillna('UnknownDestination', inplace = True)
    
    # fill missing cabin data with unknown U or 0 for cabin number
    df['Cabin'].fillna('U/0/U', inplace = True)
    
    # fill nan with mode
    df['CryoSleep'].fillna(df['CryoSleep'].mode(), inplace = True)
    df['VIP'].fillna(df['VIP'].mode(), inplace = True)
    
    # fill missing data using median
    df['Age'].fillna(df['Age'].median(), inplace = True)
    df['RoomService'].fillna(df['RoomService'].median(), inplace = True)
    df['VRDeck'].fillna(df['VRDeck'].median(), inplace = True)
    df['Spa'].fillna(df['Spa'].median(), inplace = True)
    df['ShoppingMall'].fillna(df['ShoppingMall'].median(), inplace = True)
    df['FoodCourt'].fillna(df['FoodCourt'].median(), inplace = True)
    return df


# In[5]:


train = fill_na_records(train)
test = fill_na_records(test)


# In[ ]:





# In[6]:


def boolean_converter(value):
    if value == True:
        return 1
    else:
        return 0
    
def cabin_number_converter(cabin):
    if isinstance(cabin, str):
        d, n, s = cabin.split("/")
        return int(n)
    
def clean_boolean(data):
    if "Transported" in data.columns.tolist():
        data["Transported"] = data["Transported"].apply(boolean_converter)
    data["VIP"] = data["VIP"].apply(boolean_converter)
    data["CryoSleep"] = data["CryoSleep"].apply(boolean_converter)
    return data

def clean_home_planet(data):
    for i in data["HomePlanet"].unique():
        if isinstance(i, str):
            data[f"HomePlanet_{i}"] = data["HomePlanet"].apply(lambda x: 1 if x == i else 0)
    data = data.drop(['HomePlanet'], axis=1)
    return data

def clean_destination(data):
    for i in data["Destination"].unique():
        if isinstance(i, str):
            data[f"Destination_{i}"] = data["Destination"].apply(lambda x: 1 if x == i else 0)
    data = data.drop(['Destination'], axis=1)
    return data

def clean_cabin(data):
    cabin_deck = set()
    cabin_side = set()
    for i in data["Cabin"]:
        if isinstance(i, str):
            dec, num, side = i.split("/")
            cabin_deck.add(dec)
            cabin_side.add(side)
            
    for i in cabin_deck:
        data[f"Cabin_deck_{i}"] = data["Cabin"].apply(lambda x: 1 if str(x).split("/")[0] == i else 0)
        
    for i in cabin_side:
        data[f"Cabin_side_{i}"] = data["Cabin"].apply(lambda x: 1 if str(x).split("/")[2] == i else 0)
        
    data["Cabin_number"] = data["Cabin"].apply(cabin_number_converter)
    data = data.drop(['Cabin'], axis=1)
    return data


# In[7]:


clean_bool = clean_boolean(train)
clean_home = clean_home_planet(train)
clean_dest = clean_destination(clean_home)
clean_cab = clean_cabin(clean_dest)
clean_cab


# In[8]:


train = clean_cab.drop(["PassengerId", "Name"], axis=1)
train.columns


# In[9]:


target = train["Transported"]
train = train.drop(["Transported"], axis=1)


# In[16]:


train.head()


# In[10]:


train = train.reindex(columns=sorted(train.columns))


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from decimal import Decimal


# In[13]:


X_test, X_train, y_test, y_train = train_test_split(train, target, test_size=0.27, random_state=18)


# In[14]:


knn = KNeighborsClassifier(n_neighbors=9, weights='distance', n_jobs=4)
knn = knn.fit(train, target)
knn_pred = knn.predict(X_test)
accuracy_knn = round(
    Decimal(
        accuracy_score(y_test, knn_pred) * 100
    ), 2
)
accuracy_knn


# In[20]:


rf = RandomForestClassifier(n_estimators=999, criterion='entropy')
rf = rf.fit(train, target)
rf_pred = rf.predict(X_test)
importances = list(rf.feature_importances_)
accuracy_rf = round(
    Decimal(
        accuracy_score(y_test, rf_pred) * 100
    ), 2
)
print(accuracy_rf)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_test.columns, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
print(feature_importances)


# In[22]:


clean_bool = clean_boolean(test)
clean_home = clean_home_planet(test)
clean_dest = clean_destination(clean_home)
clean_cab = clean_cabin(clean_dest)
clean_cab


# In[23]:


clean_cab


# In[24]:


clean_cab = clean_cab.reindex(columns=sorted(clean_cab.columns))


# In[25]:


test = clean_cab.drop(["Name", "PassengerId"], axis=1)
test.head()


# In[26]:


test.shape


# In[27]:


len(test.columns)


# In[28]:


len(train.columns)


# In[21]:


test = test.reindex(columns=sorted(test.columns))


# In[30]:


predictions = []
columns = test.columns 
for i in test.values:
    df = pd.DataFrame.from_records([i])
    df.columns = columns
    rf_pred = rf.predict(df)
    predictions.append(rf_pred)
len(predictions)


# In[31]:


predictions


# In[32]:


df_columns = ["PassengerId", "Transported"]
df = pd.DataFrame(columns=df_columns)
passenger_list = []
prediction_list = []
for i, j in enumerate(predictions):
    passenderId = clean_cab.iloc[i].PassengerId
    pid = str(passenderId)
    passenger_list.append(pid)
    prediction = False
    if j[0] == 1:
        prediction = True
    prediction_list.append(prediction)


# In[33]:


df["PassengerId"] = passenger_list


# In[34]:


df["Transported"] = prediction_list


# In[35]:


df.to_csv('submission.csv', index=False , header = 1)


# In[36]:


df


# In[ ]:




