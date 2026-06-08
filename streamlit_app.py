import streamlit as st
import numpy as np
import json
import base64
import time

st.markdown("<body style ='color:#E2E0D9;'></body>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: #1B9E91;'>House Price Prediction in Indore, MP, India</h4>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #1B9E91;'>Estimate the price range of houses in Indore based on area, BHK, bathrooms, and location. Built using a lightweight Scikit-Learn/NumPy model.</h5>", unsafe_allow_html=True)

# 1. Load the lightweight numpy model
with open('model_files/indian_model.json', 'r') as f:
    model_data = json.load(f)

weights = np.array(model_data['weights'])
top_locations = model_data['top_locations']
std_err = model_data['std_err']

# 2. Define Indore location list and map to top dataset locations
indore_locations = [
    "Vijay Nagar",
    "Bypass Road",
    "Nipania",
    "Rajendra Nagar",
    "Mahalaxmi Nagar",
    "Bhanwarkuan",
    "Sudama Nagar",
    "Palasia",
    "Scheme No 54",
    "Annapurna Road",
    "Kanadia Road",
    "Bypass Road Phase II",
    "LIG Colony",
    "Scheme No 78",
    "Nipania Phase II",
    "Saket",
    "Rau",
    "Pipliyahana",
    "Bengali Square",
    "Nipania Phase I"
]
indore_to_bengaluru = dict(zip(indore_locations, top_locations))

# 3. Create the input fields in the sidebar
with st.sidebar:
    st.header("Property Specifications")
    
    selected_indore_loc = st.selectbox(
        "Select Location in Indore:",
        options=indore_locations + ["Other"],
        index=0
    )
    
    sqft = st.slider("Total Area (Square Feet):", min_value=300, max_value=10000, value=1200, step=50)
    bhk = st.slider("Number of Bedrooms (BHK):", min_value=1, max_value=10, value=2, step=1)
    bath = st.slider("Number of Bathrooms:", min_value=1, max_value=10, value=2, step=1)
    balcony = st.slider("Number of Balconies:", min_value=0, max_value=5, value=1, step=1)
    
    st.write("[Source Dataset: Bengaluru Housing Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)")

# Helper function to format prices into Lakhs/Crores
def format_rupees(lakhs_val):
    if lakhs_val <= 0:
        lakhs_val = 5.0  # Minimum floor price of 5 Lakhs
    if lakhs_val >= 100.0:
        crores_val = lakhs_val / 100.0
        return f"Rs. {crores_val:,.2f} Crores"
    else:
        return f"Rs. {lakhs_val:,.2f} Lakhs"

# 4. Calculation Trigger
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3:
    center_button = st.button('Calculate range of house price')

if center_button:
    with st.spinner('Calculating....'):
        time.sleep(1.5)

    # Construct the feature vector
    selected_bengaluru_loc = indore_to_bengaluru.get(selected_indore_loc, "Other")
    x = [
        1.0,  # intercept
        float(bhk),
        float(sqft),
        float(bath),
        float(balcony)
    ]
    for top_loc in top_locations:
        x.append(1.0 if selected_bengaluru_loc == top_loc else 0.0)
    
    # Calculate price prediction (in Lakhs)
    pred_lakhs = np.dot(x, weights)
    lower_lakhs = pred_lakhs - 1.96 * std_err
    higher_lakhs = pred_lakhs + 1.96 * std_err
    
    # Display Results
    st.markdown("<h5 style='text-align: center; color: #1B9E91;'>The price range of your house is between:</h5>", unsafe_allow_html=True)
    
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.write("")
    with col_res2:
        st.subheader(format_rupees(lower_lakhs))
        st.subheader("       AND ")
        st.subheader(format_rupees(higher_lakhs))
    with col_res3:
        st.write("")

    # Display base64 cat/kramer GIF
    try:
        file_ = open("kramer_gif.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.markdown(
            f'<center><img src="data:image/gif;base64,{data_url}" alt="cat gif"></center>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        pass
