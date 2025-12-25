import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from attention_utils import extract_attention

st.set_page_config(page_title="AI Review Analyzer", layout="wide")


@st.cache_resource
def load_model():
    device = torch.device("cpu")  # Streamlit Cloud is CPU-only

    tokenizer = BertTokenizer.from_pretrained("Agr07/bert_fake")
    model = BertForSequenceClassification.from_pretrained(
        "Agr07/bert_fake",
        output_attentions=True
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device


model, tokenizer, device = load_model()

st.title("🛒 AI Product Review Quality Analyzer")

review = st.text_area("Paste product review here", height=150)

if st.button("Analyze Review"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        inputs = tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        fake_prob = torch.softmax(outputs.logits, dim=1)[0][1].item()

        st.metric("Fake Review Probability", f"{fake_prob:.2%}")

        if fake_prob > 0.6:
            st.error("🔴 Likely Fake Review")
        else:
            st.success("🟢 Likely Genuine Review")

        st.subheader("🔍 Attention Visualization")

        token_scores = extract_attention(
            model, tokenizer, review, device
        )

        html = ""
        for token, score in token_scores:
            if token.startswith("##"):
                continue
            html += (
                f"<span style='background-color:rgba(255,0,0,{score})'>"
                f"{token} </span>"
            )

        st.markdown(html, unsafe_allow_html=True)



