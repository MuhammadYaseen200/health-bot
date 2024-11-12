import streamlit as st
import os
from transformers import pipeline

# Set the Hugging Face token directly (replace this with your token)
hf_token = "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"

# Initialize the Hugging Face model pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", use_auth_token=hf_token)


# Step 2: Initialize the Hugging Face model pipeline
try:
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", use_auth_token=hf_token)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Step 3: Define the Symptom Analysis Function
def analyze_symptoms(symptoms):
    """Analyzes symptoms input by the user and provides general health guidance."""
    possible_questions = [
        "What could be causing these symptoms?",
        "Should I see a doctor for these symptoms?",
        "Are there any home remedies or treatments for these symptoms?",
    ]

    responses = []
    for question in possible_questions:
        try:
            response = qa_pipeline({'question': question, 'context': symptoms})
            responses.append(f"{question} -> {response['answer']}")
        except Exception as e:
            responses.append(f"Error processing question: {question} - {str(e)}")

    return "\n\n".join(responses)

# Step 4: Set Up Streamlit Interface
st.title("Symptom Checker - AI Guidance")
st.write("Enter your symptoms, and the AI will suggest possible causes and advise whether to see a doctor.")

# Text input for user symptoms
symptoms = st.text_input("Enter your symptoms:")

# Button to run the analysis
if st.button("Analyze Symptoms"):
    if symptoms:
        response = analyze_symptoms(symptoms)
        st.subheader("AI Response")
        st.write(response)
    else:
        st.warning("Please enter some symptoms to analyze.")
