import streamlit as st
import requests

st.title("AI Ticket Triage Dashboard")
st.write("Paste a support ticket below to see predicted category, urgency, and routing decision.")

ticket_text = st.text_area("Ticket text", height=150)

if st.button("Classify Ticket"):
    if ticket_text.strip():
        response = requests.post("http://127.0.0.1:8000/predict", json={"text": ticket_text})
        result = response.json()

        col1, col2 = st.columns(2)
        col1.metric("Category", result["category"], f"{result['category_confidence']*100:.1f}% confidence")
        col2.metric("Urgency", result["urgency"], f"{result['urgency_confidence']*100:.1f}% confidence")

        if result["needs_human_review"]:
            st.warning("⚠️ Low confidence — flagged for human review")
        else:
            st.success("✅ Auto-routed")
    else:
        st.error("Please enter ticket text.")