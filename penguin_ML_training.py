import streamlit as st
import pandas as pd
penguins = pd.read_csv("penguins_cleaned.csv")
df = penguins.copy() # create a new data which is the same as data "penguins"
target = 'species' # certify the target data which is "species"

# Encoding processes of "sex" and  "island"
encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix = col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

# Converting "species" including "Adelie", "Chinstrap", and "Gentoo" to values of 0, 1, 2
target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2} # data "species" has "Adelie", "Chinstrap", and "Gentoo"
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)
st.write(df)

# Separating X and y
X = df.drop('species', axis=1)  # Remove column of "species"
Y = df['species']
#st.write(X)

# Packages of building some machine learning models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# Save the model
import pickle

# Build and save Random Forest Classifier model
clf = RandomForestClassifier()
clf.fit(X,Y)
pickle.dump(clf, open('penguins_rf_clf.pkl', 'wb'))

# Build and save Logistic Regression model
clf_lr = LogisticRegression(max_iter=1000)
clf_lr.fit(X, Y)
pickle.dump(clf_lr, open('penguins_lr_clf.pkl', 'wb'))

# Build and save Support Vector Classifier model
clf_svc = SVC(probability=True)
clf_svc.fit(X, Y)
pickle.dump(clf_svc, open('penguins_svc_clf.pkl', 'wb'))

# Build and save K-Neighbors Classifier model
clf_knn = KNeighborsClassifier()
clf_knn.fit(X, Y)
pickle.dump(clf_knn, open('penguins_knn_clf.pkl', 'wb'))

# Build and save Gradient Boosting Classifier model
clf_gb = GradientBoostingClassifier()
clf_gb.fit(X, Y)
pickle.dump(clf_gb, open('penguins_gb_clf.pkl', 'wb'))

# Display the final DataFrame in Streamlit
st.write("Data after encoding.")
st.write(df)

st.write("Models have been trained and saved.")