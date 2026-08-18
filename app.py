import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL AND DATA
# ---------------------------------------------------

try:
    model = joblib.load("car_price_model.pkl")
    df = pd.read_csv("cardekho_dataset.csv")

except Exception as e:
    st.error("Unable to load the model or dataset.")
    st.error(str(e))
    st.stop()

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🚗 Car Price Prediction System")

st.write(
    "Enter the details of a used car to estimate its "
    "selling price using Machine Learning."
)

st.divider()

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

st.subheader("🚘 Enter Car Details")

col1, col2 = st.columns(2)

with col1:

    brand = st.selectbox(
        "🏷️ Car Brand",
        sorted(df["brand"].dropna().unique())
    )

    model_name = st.selectbox(
        "🚘 Car Model",
        sorted(
            df[df["brand"] == brand]["model"]
            .dropna()
            .unique()
        )
    )

    vehicle_age = st.number_input(
        "📅 Vehicle Age (Years)",
        min_value=0,
        max_value=50,
        value=3
    )

    km_driven = st.number_input(
        "🛣️ Kilometers Driven",
        min_value=0,
        value=30000,
        step=1000
    )

    mileage = st.number_input(
        "⛽ Mileage",
        min_value=0.0,
        value=18.0,
        step=0.1
    )

    engine = st.number_input(
        "🔧 Engine (CC)",
        min_value=0.0,
        value=1200.0,
        step=100.0
    )


with col2:

    max_power = st.number_input(
        "⚡ Max Power (bhp)",
        min_value=0.0,
        value=85.0,
        step=1.0
    )

    seats = st.number_input(
        "💺 Number of Seats",
        min_value=1,
        max_value=20,
        value=5
    )

    seller_type = st.selectbox(
        "👤 Seller Type",
        sorted(df["seller_type"].dropna().unique())
    )

    fuel_type = st.selectbox(
        "⛽ Fuel Type",
        sorted(df["fuel_type"].dropna().unique())
    )

    transmission_type = st.selectbox(
        "⚙️ Transmission Type",
        sorted(df["transmission_type"].dropna().unique())
    )

st.divider()

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if st.button(
    "🔮 Predict Car Price",
    use_container_width=True
):

    try:

        # Create user input
        input_data = pd.DataFrame({

            # Required because the trained model
            # contains this feature
            "Unnamed: 0": [0],

            "vehicle_age": [vehicle_age],

            "km_driven": [km_driven],

            "mileage": [mileage],

            "engine": [engine],

            "max_power": [max_power],

            "seats": [seats],

            "brand": [brand],

            "model": [model_name],

            "seller_type": [seller_type],

            "fuel_type": [fuel_type],

            "transmission_type": [transmission_type]
        })

        # ------------------------------------------------
        # ONE-HOT ENCODING
        # ------------------------------------------------

        input_data = pd.get_dummies(
            input_data,
            columns=[
                "brand",
                "model",
                "seller_type",
                "fuel_type",
                "transmission_type"
            ],
            drop_first=True
        )

        # ------------------------------------------------
        # GET EXACT FEATURES USED BY THE MODEL
        # ------------------------------------------------

        if hasattr(model, "feature_names_in_"):

            expected_features = list(
                model.feature_names_in_
            )

        else:

            try:
                expected_features = list(
                    joblib.load("feature_names.pkl")
                )

            except Exception:
                st.error(
                    "Feature names could not be loaded."
                )
                st.stop()

        # ------------------------------------------------
        # MATCH FEATURES
        # ------------------------------------------------

        input_data = input_data.reindex(
            columns=expected_features,
            fill_value=0
        )

        # ------------------------------------------------
        # MAKE PREDICTION
        # ------------------------------------------------

        prediction = model.predict(input_data)[0]

        # ------------------------------------------------
        # DISPLAY RESULT
        # ------------------------------------------------

        st.success("✅ Prediction completed successfully!")

        st.metric(
            label="🚗 Estimated Car Selling Price",
            value=f"₹ {prediction:,.0f}"
        )

        st.info(
            "ℹ️ Note: This is an estimated price generated "
            "by the Machine Learning model and may differ "
            "from the actual market price."
        )

    except Exception as e:

        st.error("❌ Prediction could not be completed.")

        st.code(str(e))


# ---------------------------------------------------
# ABOUT PROJECT
# ---------------------------------------------------

st.divider()

with st.expander("ℹ️ About This Project"):

    st.write(
        """
        This Car Price Prediction system uses Machine Learning
        to estimate the selling price of a used car.

        The prediction is based on factors such as:

        • Car brand and model
        • Vehicle age
        • Kilometers driven
        • Fuel type
        • Transmission type
        • Mileage
        • Engine capacity
        • Maximum power
        • Number of seats
        • Seller type

        The system is developed as an MCA PBL project using
        Python, Machine Learning, Pandas, Scikit-learn and Streamlit.
        """
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.caption(
    "Car Price Prediction System | Machine Learning PBL Project"
)