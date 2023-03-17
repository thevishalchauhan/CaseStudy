import streamlit as st
import lightgbm as lgb
import pandas as pd
import numpy as np
import time

st.title("Titanic Model Prediction")

with st.form("my_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = float(st.slider("Select Age",min_value=0,max_value=100))
        # st.write("Age:",age)
        destination = st.selectbox(
            'Destination',
            ('55 Cancri e', 'PSO J318.5-22', 'TRAPPIST-1e'))
        home = st.selectbox(
            'Select Home',
            ('Earth', 'Europa', 'Mars'))
    
        cyro_sleep = st.radio(
            'CryoSleep',
            (True, False))
        vip = st.radio(
            'VIP',
            (True, False))
    with col2:
        room_sevice = float(st.slider("Room Service Range",min_value=0,max_value=15000))
        # st.write("Room Serivce:",room_sevice)
        food_court = float(st.slider("FoodCourt Range",min_value=0,max_value=30000))
        # st.write("FoodCourt:",food_court)
        shopping_mall = float(st.slider("ShoppingMall Range",min_value=0,max_value=25000))
        # st.write("ShoppingMall:",shopping_mall)
        spa = float(st.slider("Spa Range",min_value=0,max_value=25000))
        # st.write("Spa:",spa)
        vr_deck = float(st.slider("VRDeck Range",min_value=0,max_value=25000))
        # st.write("VRDeck:",vr_deck)

    submitted = st.form_submit_button("Submit")
    if submitted:
        data = {
            'Age': age,
            'RoomService': room_sevice,
            'FoodCourt': food_court,
            'ShoppingMall': shopping_mall,
            'Spa': spa,
            'VRDeck':vr_deck
       }
        #Cyro Sleep
        if cyro_sleep:
            data['CryoSleep_True'] = 1
            data['CryoSleep_False'] = 0
        else:
            data['CryoSleep_True'] = 0
            data['CryoSleep_False'] = 1

        # VIP
        if vip:
            data['VIP_True'] = 1
            data['VIP_False'] = 0
        else:
            data['VIP_True'] = 0
            data['VIP_False'] = 1
        
        # Destination
        if destination == 'TRAPPIST-1e':
            data['Destination_TRAPPIST-1e'] = 1
            data['Destination_PSO J318.5-22'] = 0
            data['Destination_55 Cancri e'] = 0
        elif destination == 'PSO J318.5-22':
            data['Destination_TRAPPIST-1e'] = 0
            data['Destination_PSO J318.5-22'] = 1
            data['Destination_55 Cancri e'] = 0
        else:
            data['Destination_TRAPPIST-1e'] = 0
            data['Destination_PSO J318.5-22'] = 0
            data['Destination_55 Cancri e'] = 1
        # Home Planet
        if home == 'Earth':
            data['HomePlanet_Earth'] = 1
            data['HomePlanet_Europa'] = 0
            data['HomePlanet_Mars'] = 0
        elif home == 'Europa':
            data['HomePlanet_Earth'] = 0
            data['HomePlanet_Europa'] = 1
            data['HomePlanet_Mars'] = 0
        else:
            data['HomePlanet_Earth'] = 0
            data['HomePlanet_Europa'] = 0
            data['HomePlanet_Mars'] = 1

        df = pd.DataFrame(data,index=[0])
        clf = lgb.Booster(model_file='weights/lgbr_base.txt')
        pred = clf.predict(df)
        # st.write("Data",data)
        result = np.where(pred > 0.5, True, False)
        st.subheader(f'[Prediction] Transported: {result[0]}')