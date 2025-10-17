import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="NLP HUB", page_icon="</>", layout="wide")
st.title("WELCOME TO MY FIRST TRANSFORMER APP")

task = st.selectbox(
    "Select something from below to proceed",
    [
        "Sentiment Analysis",
        "Named Entity Recognition",
        "Zero-Shot Classification",
        "Summarization",
        "Question Answering",
        "Text Generation",
    ],
)

main_text = st.text_area("ENTER YOUR TEXT")

ans = "Nothing here"

if st.button("RUN"):
    
    if len(main_text.strip()) == 0:
        st.warning("⚠️ Please enter some text first!")  
    else:
        if task == "Sentiment Analysis":
            model = pipeline("sentiment-analysis")
            ans = model(main_text)

        elif task == "Summarization":
            model = pipeline("summarization", model="facebook/bart-large-cnn")
            ans = model(main_text, max_length=120, min_length=40, do_sample=False)

        elif task == "Text Generation":
            model = pipeline("text-generation", model="gpt2")
            ans = model(main_text, max_length=80, num_return_sequences=1)

        elif task == "Named Entity Recognition":
            model = pipeline("ner", grouped_entities=True, model="dslim/bert-base-NER")
            ans = model(main_text)

        elif task == "Question Answering":
         
            q = st.text_input("Type your question here:")
            if len(q.strip()) == 0:
                st.warning("⚠️ Please enter a question first!")
            else:
                model = pipeline("question-answering", model="deepset/roberta-base-squad2")
                ans = model(question=q, context=main_text)

        elif task == "Zero-Shot Classification":
           
            labels_raw = st.text_input("Enter comma-separated labels:")
            if len(labels_raw.strip()) == 0:
                st.warning("⚠️ Please enter some labels!")
            else:
                labels = [label.strip() for label in labels_raw.split(",")]
                model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
                ans = model(main_text, candidate_labels=labels)

st.subheader("Result")
st.json(ans)
