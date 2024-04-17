import streamlit as st
import pandas as pd
import pickle

# Load the machine learning model
with open('artifacts\model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the preprocessing pipeline
with open('artifacts\preprocessor.pkl', 'rb') as f:
    preprocessing_pipeline = pickle.load(f)

# Define the function to predict customer satisfaction
def predict_satisfaction(input_data):
    preprocessed_data = preprocessing_pipeline.transform(input_data)
    prediction = model.predict(preprocessed_data)
    return prediction

# Main function to run the Streamlit app
def main():
    st.title('Customer Satisfaction Prediction')

    # Input form for user to enter parameters
    st.sidebar.header('Input Parameters')
    id = st.sidebar.number_input('ID', min_value=0)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    customer_type = st.sidebar.selectbox('Customer Type', ['Loyal Customer', 'Disloyal Customer'])
    age = st.sidebar.number_input('Age', min_value=0)
    type_of_travel = st.sidebar.selectbox('Type of Travel', ['Personal Travel', 'Business travel'])
    class_type = st.sidebar.selectbox('Class', ['Eco Plus', 'Business', 'Eco'])
    flight_distance = st.sidebar.number_input('flight_distance', min_value=0)
    departure_delay_in_minutes = st.sidebar.number_input('departure_delay_in_minutes', min_value=0)
    arrival_delay_in_minutes = st.sidebar.number_input('arrival_delay_in_minutes', min_value=0)
    # Ratings for various services
    inflight_wifi_service = st.sidebar.slider('Inflight WiFi Service', 0, 5, 3)
    departure_arrival_time_convenient = st.sidebar.slider('Departure/Arrival Time Convenience', 0, 5, 3)
    ease_of_online_booking = st.sidebar.slider('Ease of Online Booking', 0, 5, 3)
    gate_location =st.sidebar.slider('gate_location', 0, 5, 3)
    food_and_drink = st.sidebar.slider('food_and_drink', 0, 5, 3)
    online_boarding = st.sidebar.slider('online_boarding', 0, 5, 3)
    seat_comfort = st.sidebar.slider('seat_comfort', 0, 5, 3)
    inflight_entertainment = st.sidebar.slider('inflight_entertainment', 0, 5, 3)
    on_board_service = st.sidebar.slider('on_board_service', 0, 5, 3)
    leg_room_service = st.sidebar.slider('leg_room_service', 0, 5, 3)
    baggage_handling = st.sidebar.slider('baggage_handling', 0, 5, 3)
    checkin_service = st.sidebar.slider('checkin_service', 0, 5, 3)
    inflight_service = st.sidebar.slider('inflight_service', 0, 5, 3)
    cleanliness = st.sidebar.slider('cleanliness', 0, 5, 3)

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'id': [id],
        'gender': [gender],
        'customer_type': [customer_type],
        'age': [age],
        'type_of_travel': [type_of_travel],
        'Class': [class_type],
        'flight_distance': [flight_distance],
        'inflight_wifi_service': [inflight_wifi_service],
        'departure_arrival_time_convenient': [departure_arrival_time_convenient],
        'ease_of_online_booking': [ease_of_online_booking],
        'gate_location': [gate_location],
        'food_and_drink': [food_and_drink],
        'online_boarding': [online_boarding],
        'seat_comfort': [seat_comfort],
        'inflight_entertainment': [inflight_entertainment],
        'on_board_service': [on_board_service],
        'leg_room_service': [leg_room_service],
        'baggage_handling': [baggage_handling],
        'checkin_service': [checkin_service],
        'inflight_service': [inflight_service],
        'cleanliness': [cleanliness],
        'departure_delay_in_minutes': [departure_delay_in_minutes],
        'arrival_delay_in_minutes': [arrival_delay_in_minutes]
    })

    # Display input data
    st.subheader('Input Data')
    st.write(input_data)

    # Predict customer satisfaction
    if st.sidebar.button('Predict'):
        prediction = predict_satisfaction(input_data)
        st.subheader('Prediction')
        st.write(prediction)

if __name__ == '__main__':
    main()
