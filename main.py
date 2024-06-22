import streamlit as st

import streamlit as st

st.page_link("main.py", label="Home", icon="🏠")
st.page_link("pages/page3.py", label="Page 1", icon="1️⃣")
st.page_link("pages/page_2.py", label="Page 2", icon="2️⃣", disabled=True)
st.page_link("http://www.google.com", label="Google", icon="🌎")