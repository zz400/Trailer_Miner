#!/usr/bin/env python

"""
Test Streamlit

Author: Zhengyang Zhao
Date: Jun. 13 2020.
"""

import os
import time 
import sys
import random
import numpy as np
import pandas as pd

from textblob import TextBlob
from PIL import Image
import webbrowser
from bokeh.models.widgets import Div
import streamlit as st


# st.write('Hello, world!')
# st.write('Hello, *World!* :sunglasses:')
# st.write(1234)
# st.write("1234")
# st.header("Header")
# st.subheader("SubHeader")
# st.text("Text")
# st.text("Text1\nText2")
# st.markdown('Streamlit is **_really_ cool**.')

# Insert image:
# st.image("./image0.png", use_column_width=True)

# Image Link:
# link_md = "[![](image0.png)](https://streamlit.io)"
# st.markdown(link_md)


# Example:

# x = st.slider('Select a value')
# st.write(x, 'squared is', x * x)

# progress_bar = st.progress(0)
# status_text = st.empty()
# last_rows = np.random.randn(1, 1)
# chart = st.line_chart(last_rows)

# for i in range(1, 101):
#     new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
#     status_text.text("%i%% Complete" % i)
#     chart.add_rows(new_rows)
#     progress_bar.progress(i)
#     last_rows = new_rows
#     time.sleep(0.001)

# # progress_bar.empty()

# chart_data = pd.DataFrame(
#                 np.random.randn(20, 3),
#                 columns=['a', 'b', 'c'])
# st.line_chart(chart_data)


# st.title("Welcome to Trailer Miner!")

# st.subheader("Please input a movie title:")



html = """
  <style>
    /* Disable overlay (fullscreen mode) buttons */
    .overlayBtn {
      display: none;
    }

    /* Remove horizontal scroll */
    .element-container {
      width: auto !important;
    }

    .fullScreenFrame > div {
      width: auto !important;
    }

    /* 2nd thumbnail */
    .element-container:nth-child(4) {
      top: -266px;
      left: 350px;
    }

    /* 1st button */
    .element-container:nth-child(3) {
      left: 10px;
      top: -60px;
    }

    /* 2nd button */
    .element-container:nth-child(5) {
      left: 360px;
      top: -326px;
    }
  </style>
"""
# st.markdown(html, unsafe_allow_html=True)

# st.image("https://www.w3schools.com/howto/img_forest.jpg", width=300)
# st.button("Show", key=1)

# st.image("https://www.w3schools.com/howto/img_forest.jpg", width=300)
# st.button("Show", key=2)

image1 = "https://www.w3schools.com/howto/img_forest.jpg"
image2 = "https://www.w3schools.com/howto/img_forest.jpg"

href1 = "https://docs.streamlit.io/"
caption1 = "Trailer 1"

bbb = """
        <table><tr>
        <td> <a href={} target="_blank"><img src={} alt="Drawing" style="width: 200px;"/><figcaption>Fig.1</figcaption> </td>
        <td> <img src={} alt="Drawing" style="width: 200px;"/> </td>
        </tr></table>
        """.format(href1, image1, image2,)
st.markdown(bbb, unsafe_allow_html=True)
st.write(bbb)