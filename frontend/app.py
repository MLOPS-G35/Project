import streamlit as st
import requests

URL="http://127.0.0.1:8000"

st.title("My FastAPI App")

if st.button("Ping backend"):
    r = requests.get(f"{URL}/")
    st.json(r.json())
