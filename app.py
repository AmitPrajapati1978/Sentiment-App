#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 16:40:48 2022

@author: amit
"""

import streamlit as st
import pickle

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

st.header('Lets Check The Sentiment of the Review!(😭,😍)')


Text=st.text_input('Enter in your Text')

file = open("pickle_model.pkl", 'rb')
pickle_model = pickle.load(file)


file = open("feature.pkl", 'rb')
vocab = pickle.load(file)

col1, col2, col3 , col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3 :
    center_button = st.button('Lets Check 🤔','Your Text')
html_string = "<marquee>Click the button to predict.</marquee>"

st.markdown(html_string, unsafe_allow_html=True)


if center_button:
    if len(Text)>0:
        transformer = TfidfTransformer()
        loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
        vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([Text]))
        
        pred =pickle_model.predict(vectorised_review)
        
        
        if (pred[0] == 0):
                    st.error("The Review is Negative 😭")
                    
                    
        
        
        
        elif (pred[0] == 1 ):
                    st.success("The Review is Positive 😍")
    else:
        st.error('Please give some Input')




