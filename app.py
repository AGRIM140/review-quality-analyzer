import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from attention_utils import extract_attention

st.set_page_config(
    page_title="AI Review Analyzer",
    page_icon="🛒",
    layout="wide"
)

# -------------------- MODEL LOADING --------------------

@st.cache_resource
def load_model():
    device = torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained("Agr07/bert_fake")
    model = BertForSequenceClassification.from_pretrained(
        "Agr07/bert_fake",
        output_attentions=True
    )

    model.to(device)
    model.eval()
    return model, tokenizer, device


model, tokenizer, device = load_model()

# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("## 🧠 AI Review Analyzer")
    st.markdown(
        """
        **What this app does**
        - Detects potentially **fake product reviews**
        - Uses a **fine-tuned BERT model**
        - Trained on **Amazon Reviews Dataset**
        """
    )
    st.markdown("---")
    st.markdown("**Model:** BERT (Fine-tuned)")
    st.markdown("**Platform:** Streamlit Cloud")
    st.markdown("**Author:** Agrim Singh")

# -------------------- MAIN UI --------------------

st.markdown(
    """
    <h1 style='margin-bottom: 0;'>🛒 AI Product Review Quality Analyzer</h1>
    <p style='color: gray; margin-top: 5px;'>
        Analyze product reviews using a fine-tuned BERT model to detect fake or low-quality reviews.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("### ✍️ Enter a product review")

review = st.text_area(
    "",
    placeholder="Example: This product is amazing! I loved it so much...",
    height=160
)

analyze = st.button("🔍 Analyze Review", use_container_width=True)

# -------------------- PREDICTION --------------------

if analyze:
    if not review.strip():
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing review..."):
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

        st.markdown("---")
        st.markdown("### 📊 Analysis Result")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(
                label="Fake Review Probability",
                value=f"{fake_prob:.2%}"
            )

        with col2:
            st.progress(fake_prob)

        if fake_prob > 0.6:
            st.error("🚨 **Likely Fake Review**")
        elif fake_prob > 0.4:
            st.warning("⚠️ **Suspicious Review**")
        else:
            st.success("✅ **Likely Genuine Review**")

        # -------------------- ATTENTION VISUALIZATION --------------------

        st.markdown("---")
        st.markdown("### 🔍 Attention-Based Word Importance")
        st.caption(
            "Words highlighted darker contributed more strongly to the model's decision."
        )

        token_scores = extract_attention(model, tokenizer, review, device)

        html = ""
        for token, score in token_scores:
            if token.startswith("##"):
                continue
            intensity = min(max(score, 0.05), 0.9)
            html += (
                f"<span style='background-color:rgba(255, 87, 87, {intensity}); "
                f"padding:2px 4px; margin:2px; border-radius:4px;'>"
                f"{token}</span> "
            )

        st.markdown(
            f"<div style='line-height: 2; font-size: 16px;'>{html}</div>",
            unsafe_allow_html=True
        )

# -------------------- FOOTER --------------------

st.markdown("---")
st.caption(
    "⚙️ Built using Streamlit & Hugging Face Transformers | "
    "Fine-tuned BERT for NLP-based review quality analysis"
)


