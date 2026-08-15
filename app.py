
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

model = keras.models.load_model(
    "delivery_time_neural_network.keras"
)

preprocessor = joblib.load(
    "delivery_time_preprocessor.pkl"
)

st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="🚚"
)

st.title("🚚 Delivery Time Prediction")
st.write("Neural Network based delivery time prediction")

market_id = st.number_input(
    "Market ID",
    min_value=0,
    value=1
)

total_items = st.number_input(
    "Total Items",
    min_value=1,
    value=3
)

subtotal = st.number_input(
    "Subtotal",
    min_value=0.0,
    value=2500.0
)

num_distinct_items = st.number_input(
    "Number of Distinct Items",
    min_value=1,
    value=3
)

min_item_price = st.number_input(
    "Minimum Item Price",
    min_value=0.0,
    value=500.0
)

max_item_price = st.number_input(
    "Maximum Item Price",
    min_value=0.0,
    value=1200.0
)

total_onshift_partners = st.number_input(
    "On-shift Partners",
    min_value=0,
    value=50
)

total_busy_partners = st.number_input(
    "Busy Partners",
    min_value=0,
    value=30
)

total_outstanding_orders = st.number_input(
    "Outstanding Orders",
    min_value=0,
    value=20
)

order_hour = st.slider(
    "Order Hour",
    0,
    23,
    19
)

order_day_of_week = st.slider(
    "Day of Week (0=Monday, 6=Sunday)",
    0,
    6,
    5
)

order_month = st.slider(
    "Order Month",
    1,
    12,
    8
)

store_frequency = st.number_input(
    "Store Frequency",
    min_value=0.0,
    value=0.01,
    format="%.6f"
)

store_primary_category = st.text_input(
    "Restaurant Category",
    value="american"
)

order_protocol = st.number_input(
    "Order Protocol",
    min_value=0,
    value=1
)

if st.button("Predict Delivery Time"):

    input_data = pd.DataFrame([{
        "market_id": market_id,
        "total_items": total_items,
        "subtotal": subtotal,
        "num_distinct_items": num_distinct_items,
        "min_item_price": min_item_price,
        "max_item_price": max_item_price,
        "total_onshift_partners": total_onshift_partners,
        "total_busy_partners": total_busy_partners,
        "total_outstanding_orders": total_outstanding_orders,
        "order_hour": order_hour,
        "order_day_of_week": order_day_of_week,
        "order_month": order_month,
        "store_frequency": store_frequency,
        "store_primary_category": store_primary_category,
        "order_protocol": order_protocol
    }])

    processed = preprocessor.transform(input_data)

    prediction_log = model.predict(
        processed,
        verbose=0
    )[0][0]

    prediction = np.expm1(prediction_log)

    prediction = max(0, float(prediction))

    st.success(
        f"Estimated Delivery Time: {prediction:.2f} minutes"
    )
