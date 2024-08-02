import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.sidebar.header("User Input Features")
st.sidebar.markdown(
    """
    [Example: penguins_predicted.csv](https://github.com/dongnc1987/ML_machinelearning_file.git)
    """
)

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

st.subheader("User Input features")
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file) 
    st.write(input_df)
else:
    def user_input_features():
        island = st.sidebar.selectbox("Island", ("Biscoe", "Dream", "Torgersen"))
        sex = st.sidebar.selectbox("Sex", ("male", "female"))
        bill_length_mm = st.sidebar.number_input("Bill length (mm)", 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.number_input("Bill depth (mm)", 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.number_input("Flipper length (mm)", 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.number_input("Body mass (g)", 2700.0, 6300.0, 4207.0)
        data = {
            "island": island,
            "bill_length_mm": bill_length_mm,
            "bill_depth_mm": bill_depth_mm,
            "flipper_length_mm": flipper_length_mm,
            "body_mass_g": body_mass_g,
            "sex": sex
        }
        #st.write(data)
        features = pd.DataFrame(data, index=[0]) # Index = [0] means the initial row is label 0
        return features
    input_df = user_input_features()
    st.write(input_df)
# Number of rows before encoding
num_rows_before = input_df.shape[0]

penguins_raw = pd.read_csv("penguins_cleaned.csv")
#st.write(penguins_raw)
penguins = penguins_raw.drop(columns=["species"]) #columns=["species"]) means that removing column of species
#st.write(penguins)
df = pd.concat([input_df, penguins], axis=0) # insert input_df to "penguins" data, axis = 0 --> insert row

#st.write(df)

# encoding two colums of sex and island
encode = ["sex", "island"]
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)  # Encoding string data
    #st.write(dummy)
    df = pd.concat([df, dummy], axis=1) # asis = 1 --> insert column
    #st.write(df)
    del df[col] # delete the column of "sex"
    #st.write(df)
# Ensure we only keep the initial number of rows
df = df.iloc[:num_rows_before]

# if uploaded_file is not None:
#     st.write(input_df)
# else:
#     st.write("Awaiting CSV file to be uploaded.")
#     st.write(input_df)

load_rf_clf = pickle.load(open("penguins_rf_clf.pkl", "rb"))
load_lr_clf = pickle.load(open("penguins_lr_clf.pkl", "rb"))
load_svc_clf = pickle.load(open("penguins_svc_clf.pkl", "rb"))
load_knn_clf = pickle.load(open("penguins_knn_clf.pkl", "rb"))
load_gb_clf = pickle.load(open("penguins_gb_clf.pkl", "rb"))

# Create the choise of machine learning models
ml_model = st.sidebar.selectbox("ML models", ("RandomForestClassifier", "LogisticRegression", "SupportVectorClassifier", "KNeighborsClassifier", "GradienBoostingClassifier"))
#ml_model = st.selectbox("ML models", ("RandomForestClassifier", "LogisticRegression", "SupportVectorClassifier", "KNeighborsClassifier", "GradienBoostingClassifier"))
if ml_model == "RandomForestClassifier":
    prediction = load_rf_clf.predict(df)
    prediction_proba = load_rf_clf.predict_proba(df)
    penguins_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
    data_rows = []
    for i, (pred, probs) in enumerate(zip(prediction, prediction_proba)):
        row_data = {
            # "Sample": i + 1,
            "Predicted Species": penguins_species[pred],
            **{f"Probability_{species}": prob for species, prob in zip(penguins_species, probs)}
        }
        data_rows.append(row_data)

    # Create DataFrame from the list of dictionaries
    st.subheader("Prediction results by Random Forest Classification")
    probabilities_df = pd.DataFrame(data_rows)
    st.write(probabilities_df)

elif ml_model == "LogisticRegression":
    prediction = load_lr_clf.predict(df)
    prediction_proba = load_lr_clf.predict_proba(df)
    penguins_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
    data_rows = []
    for i, (pred, probs) in enumerate(zip(prediction, prediction_proba)):
        row_data = {
            # "Sample": i + 1,
            "Predicted Species": penguins_species[pred],
            **{f"Probability_{species}": prob for species, prob in zip(penguins_species, probs)}
        }
        data_rows.append(row_data)

    # Create DataFrame from the list of dictionaries
    st.subheader("Prediction results by Logistic Regression")
    probabilities_df = pd.DataFrame(data_rows)
    st.write(probabilities_df)

elif ml_model == "SupportVectorClassifier":
    prediction = load_svc_clf.predict(df)
    prediction_proba = load_svc_clf.predict_proba(df)
    penguins_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
    data_rows = []
    for i, (pred, probs) in enumerate(zip(prediction, prediction_proba)):
        row_data = {
            # "Sample": i + 1,
            "Predicted Species": penguins_species[pred],
            **{f"Probability_{species}": prob for species, prob in zip(penguins_species, probs)}
        }
        data_rows.append(row_data)

    # Create DataFrame from the list of dictionaries
    st.subheader("Prediction results by Support Vector Classification")
    probabilities_df = pd.DataFrame(data_rows)
    st.write(probabilities_df)

elif ml_model == "KNeighborsClassifier":
    prediction = load_gb_clf.predict(df)
    prediction_proba = load_gb_clf.predict_proba(df)
    penguins_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
    data_rows = []
    for i, (pred, probs) in enumerate(zip(prediction, prediction_proba)):
        row_data = {
            # "Sample": i + 1,
            "Predicted Species": penguins_species[pred],
            **{f"Probability_{species}": prob for species, prob in zip(penguins_species, probs)}
        }
        data_rows.append(row_data)

    # Create DataFrame from the list of dictionaries
    st.subheader("Prediction results by KNeighbors Classification")
    probabilities_df = pd.DataFrame(data_rows)
    st.write(probabilities_df)

else:
    prediction = load_svc_clf.predict(df)
    prediction_proba = load_svc_clf.predict_proba(df)
    penguins_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
    data_rows = []
    for i, (pred, probs) in enumerate(zip(prediction, prediction_proba)):
        row_data = {
            # "Sample": i + 1,
            "Predicted Species": penguins_species[pred],
            **{f"Probability_{species}": prob for species, prob in zip(penguins_species, probs)}
        }
        data_rows.append(row_data)

    # Create DataFrame from the list of dictionaries
    st.subheader("Prediction results by Gradiance Bossting Classification")
    probabilities_df = pd.DataFrame(data_rows)
    st.write(probabilities_df)


# Function to perform predictions
def perform_predictions(df):
    # Perform predictions for each model
    models = {
        "RandomForestClassifier": load_rf_clf,
        "LogisticRegression": load_lr_clf,
        "SupportVectorClassifier": load_svc_clf,
        "KNeighborsClassifier": load_knn_clf,
        "GradientBoostingClassifier": load_gb_clf
    }

    results = []

    for ml_model, model in models.items():
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)

        # Prepare species names for display
        penguins_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
        predicted_species = penguins_species[prediction]

        # Append results to list
        results.append({
            "ML Model": ml_model,
            "Prediction": predicted_species[0],  # Assuming single row input, so take first element
            "Prediction Probability": prediction_proba[0][prediction[0]]  # Probability of the predicted class
        })

    # Create a DataFrame from results
    results_df = pd.DataFrame(results)

    # Highlight the best model based on highest prediction probability
    best_model_idx = results_df["Prediction Probability"].idxmax()
    results_df.loc[best_model_idx, "ML Model"] = f"**{results_df.loc[best_model_idx, 'ML Model']}**"

    return results_df

# Streamlit App
st.subheader("Comparison of Predictions and Probabilities among ML Models")

# Assuming df is your input data DataFrame
# Replace df with your actual input data DataFrame
# df = pd.read_csv("path_to_your_input_data.csv")

# Perform predictions
results_df = perform_predictions(df)

# Display the results in a table
st.write(results_df)