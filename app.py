# Step 2: Import Libraries
import streamlit as st
from transformers import pipeline

# Step 3: Access Hugging Face Token from Streamlit Secrets
hf_token = st.secrets["HF_TOKENS"]

# Initialize the Hugging Face model pipeline
try:
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", use_auth_token=hf_token)
except Exception as e:
    st.error("Error loading model. Please check your Hugging Face token and model compatibility.")
    st.stop()  # Stop execution if there's an issue loading the model

# Step 4: Define the Symptom Analysis Function
def analyze_symptoms(symptoms):
    """Analyzes symptoms input by the user and provides general health guidance."""
    # Define a list of general symptom questions to feed the model
    possible_questions = [
        "What could be causing these symptoms?",
        "Should I see a doctor for these symptoms?",
        "Are there any home remedies or treatments for these symptoms?",
    ]
    
    # Generate responses for each question and combine them
    responses = []
    for question in possible_questions:
        try:
            response = qa_pipeline({'question': question, 'context': symptoms})
            responses.append(f"{question} -> {response['answer']}")
        except Exception as e:
            responses.append(f"Error processing question: {question} - {e}")
    
    return "\n\n".join(responses)

# Step 5: Streamlit Interface Setup
st.title("Symptom Checker - AI Guidance")
st.write("Enter your symptoms, and the AI will suggest possible causes and advise whether to see a doctor.")

# Text input for user symptoms
symptoms = st.text_input("Enter your symptoms:")

# Button to run the analysis
if st.button("Analyze Symptoms"):
    if symptoms:
        # Get and display the response from the AI model
        response = analyze_symptoms(symptoms)
        st.subheader("AI Response")
        st.write(response)
    else:
        st.warning("Please enter some symptoms to analyze.")

