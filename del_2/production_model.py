import joblib
import pandas as pd


def load_model():
    """
    :return: model, DataFrame
    """
    df_100 = pd.read_csv("Data/test_samples.csv", sep=";")
    model = joblib.load("Model/rfc_model.pkl")

    return model, df_100


model, df_100_rows = load_model()
# prediction on class 0 and 1
prediction = model.predict_proba(df_100_rows)

# convert to DataFrame and add column namns
predictions = pd.DataFrame(prediction, columns=["probability class 0", "probability class 1"])

# save to csv
predictions.to_csv("Data/prediction.csv", sep=";", index=False)
