import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

st.set_page_config(page_title="NYC Airbnb AI Dashboard", layout="wide")

df = pd.read_csv("AB_NYC_2019.csv")
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

value_model = pickle.load(open("airbnb_value_model.pkl", "rb"))
value_encoder = pickle.load(open("room_encoder.pkl", "rb"))

price_model = pickle.load(open("price_model.pkl", "rb"))
price_encoder = pickle.load(open("reg_encoder.pkl", "rb"))

kmeans = pickle.load(open("kmeans.pkl", "rb"))

st.markdown(
    "<h1 style='text-align:center;color:#ff4b4b;'>🏙 NYC Airbnb AI + Value Dashboard</h1>",
    unsafe_allow_html=True
)

tabs = st.tabs(["Neighbourhood Value", " AI Listing Evaluator", "Borough Insights"])


with tabs[0]:

    st.sidebar.header("Filters")

    borough = st.sidebar.multiselect(
        "Borough",
        df["neighbourhood_group"].unique(),
        default=df["neighbourhood_group"].unique()
    )

    room_type = st.sidebar.selectbox(
        "Room Type",
        df["room_type"].unique(),
        key="sidebar_room"
    )

    filtered = df[
        (df["neighbourhood_group"].isin(borough)) &
        (df["room_type"] == room_type)
    ]

    neigh = filtered.groupby(
        ["neighbourhood", "neighbourhood_group", "latitude", "longitude"]
    ).agg({
        "price": "mean",
        "availability_365": "mean",
        "number_of_reviews": "mean"
    }).reset_index()

    neigh.columns = [
        "neighbourhood", "borough", "lat", "lon",
        "avg_price", "avg_availability", "avg_reviews"
    ]

    neigh = neigh[neigh["avg_price"] > 0]

    neigh["value_score"] = (
        neigh["avg_availability"] * neigh["avg_reviews"]
    ) / neigh["avg_price"]

    st.subheader("🗺 Value Map by Neighbourhood")

    fig = px.scatter_mapbox(
        neigh,
        lat="lat",
        lon="lon",
        color="value_score",
        size="value_score",
        hover_name="neighbourhood",
        hover_data=["borough", "avg_price", "avg_reviews"],
        zoom=10,
        height=600
    )

    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    col1.subheader(" Top Value Areas")
    col1.dataframe(neigh.sort_values("value_score", ascending=False).head(10))

    col2.subheader(" Lowest Value Areas")
    col2.dataframe(neigh.sort_values("value_score").head(10))



with tabs[1]:

    st.subheader("Enter Listing Details")

    price = st.slider("Price", 10, 500, 150)
    availability = st.slider("Availability", 0, 365, 120)
    reviews = st.slider("Reviews", 0, 300, 20)
    min_nights = st.slider("Minimum Nights", 1, 30, 2)

    room = st.selectbox(
        "Room Type",
        df["room_type"].unique(),
        key="ai_room"
    )

    room_val = value_encoder.transform([room])[0]
    room_reg = price_encoder.transform([room])[0]

    if st.button("Run AI Analysis "):

        st.markdown("###  Value Classification")
        val = value_model.predict(
            [[price, availability, reviews, min_nights, room_val]]
        )[0]
        st.success("High Value Listing" if val == 1 else "Low Value / Overpriced")

        st.markdown("###  Fair Price Prediction")
        pred_price = price_model.predict(
            [[availability, reviews, min_nights, room_reg]]
        )[0]
        st.info(f"Estimated Fair Price: ${round(pred_price, 2)}")

        st.markdown("###  Market Segment")
        cluster = kmeans.predict([[price, availability, reviews]])[0]
        st.warning(f"Listing belongs to Cluster {cluster}")



with tabs[2]:

    st.subheader(" Borough Level Insights")

    borough_stats = df.groupby("neighbourhood_group").agg({
        "price": "mean",
        "availability_365": "mean",
        "number_of_reviews": "mean"
    }).reset_index()

    st.dataframe(borough_stats)

    fig2 = px.bar(
        borough_stats,
        x="neighbourhood_group",
        y="price",
        color="price",
        title="Average Price by Borough"
    )

    st.plotly_chart(fig2, use_container_width=True)


