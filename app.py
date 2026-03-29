import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Title
st.title("📈 Tesla Stock Price Prediction")

# Load model
model = load_model("model_lstm.h5")

# Upload dataset
uploaded_file = st.file_uploader("Upload TSLA Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.tail())

    # -----------------------------
    # Historical Graph
    # -----------------------------
    st.subheader("📊 Historical Stock Price")

    df['Date'] = pd.to_datetime(df['Date'])
    st.line_chart(df.set_index('Date')['Close'])

    # -----------------------------
    # Volume Graph
    # -----------------------------
    st.subheader("📊 Trading Volume")
    st.line_chart(df.set_index('Date')['Volume'])

    # -----------------------------
    # Moving Average
    # -----------------------------
    st.subheader("📊 Moving Average (Trend Analysis)")

    df['MA_50'] = df['Close'].rolling(50).mean()

    fig_ma, ax_ma = plt.subplots()
    ax_ma.plot(df['Close'], label='Close Price')
    ax_ma.plot(df['MA_50'], label='50-Day Moving Avg')
    ax_ma.legend()
    st.pyplot(fig_ma)

    # -----------------------------
    # Prediction Section
    # -----------------------------
    data = df[['Close']].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    last_60_days = scaled_data[-60:]
    X_input = last_60_days.reshape(1, -1, 1)

    # Slider
    days = st.slider("Select number of days to predict", 1, 10)

    if st.button("Predict"):

        future_predictions = []

        for i in range(days):
            pred = model.predict(X_input)
            future_predictions.append(pred[0][0])

            X_input = np.append(X_input[:,1:,:], pred.reshape(1,1,1), axis=1)

        future_predictions = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1,1)
        )

        # -----------------------------
        # Show Predictions
        # -----------------------------
        st.subheader("📈 Predicted Prices")
        st.write(future_predictions)

        # -----------------------------
        # Prediction Graph (Better)
        # -----------------------------
        fig_pred, ax_pred = plt.subplots()

        day_axis = list(range(1, len(future_predictions)+1))

        ax_pred.plot(day_axis, future_predictions, marker='o', label='Predicted Prices')
        ax_pred.set_title("Future Stock Price Prediction")
        ax_pred.set_xlabel("Days")
        ax_pred.set_ylabel("Price")
        ax_pred.legend()

        st.pyplot(fig_pred)

        # -----------------------------
        # Day-wise Output
        # -----------------------------
        for i, price in enumerate(future_predictions):
            st.write(f"Day {i+1}: ₹{price[0]:.2f}")

