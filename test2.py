import streamlit as st
import pandas as pd

st.title("Test App")
data = {"col1": [1,2,3], "col2": [4,5,6]}
df = pd.DataFrame(data)
st.write(df)
