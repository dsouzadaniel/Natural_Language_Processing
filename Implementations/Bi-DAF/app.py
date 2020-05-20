# External Libraries
import torch
import streamlit as st

# Load the Model and PreTrained Weights
from model import architecture
from utils import helper

# Default Args
default_context = "The food was great. The service was quick and the staff was very polite."
default_query = "how was the service ?"


@st.cache(allow_output_mutation=True)
def load_model_for_app():
    # Model Definition contains default params
    BIDAF = architecture.BiDAF()
    # Pretrained Weights
    BIDAF.load_state_dict(torch.load('BIDAF.pth', map_location=torch.device('cpu')))
    BIDAF.eval()
    return BIDAF


pretrained_BIDAF = load_model_for_app()

st.title('Bi-Directional Attn Flow(BiDAF) Demo')

st.header('Input')
context_text = st.text_input(label='Enter Context Here',
                             value=default_context)
query_text = st.text_input(label='Enter Query Here',
                           value=default_query)

if len(context_text) == 0 or len(query_text) == 0:
    context_text = default_context
    query_text = default_query

highlighted_context, confidence = helper.predict(context_text, query_text, pretrained_BIDAF)

st.subheader('Predicted Span')
st.write("Answering with Confidence {0}".format(round(confidence,3)))
st.markdown(highlighted_context, unsafe_allow_html=True)
