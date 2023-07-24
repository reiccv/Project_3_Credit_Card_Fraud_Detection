import streamlit as st
import pandas as pd
import h5py
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder



# Load the model from the h5 file
model = load_model('nn_model\Credit_card_nn_model.h5')


def preprocess_data(df):
    
    label_encoder = LabelEncoder()
    df['Use Chip'] = label_encoder.fit_transform(df['Use Chip'])

    return df

def main():
    st.title('Fraud Detection App')

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV file:")
        st.write(df)

     
        df = preprocess_data(df)

        # Load the model 
        nn = RandomForestClassifier()

        X_train = pd.DataFrame({
              "Amount",
              "Year",
              "Month",
              "Day",
              "Time",
              "Use Chip",
              "Merchant Name",
              "Merchant City",
              "Merchant State",
              "Zip",
              "MCC",
          })


        y_train = pd.Series([0, 0, 0, 0, ...])  # 0 represents non-fraudulent, 1 represents fraudulent

        # Fit the model on the training data
        model.fit(X_train, y_train)

        
        X = df.drop(['Is Fraud?'], axis=1)

        # Make predictions
        y_pred = nn.predict(X)
        df['Predicted'] = ['Fraudulent' if pred == 1 else 'Not Fraudulent' for pred in y_pred]

        st.write("Predicted Results:")
        st.write(df)

if __name__ == '__main__':
    main()
