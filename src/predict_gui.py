import streamlit as st
import torch
from transformers import AutoTokenizer
from enhanced_model import EnhancedPersianSentimentModel
from enhanced_preprocessing import EnhancedPersianPreprocessor
import os
import numpy as np
import random
import pandas as pd

MODEL_PATH = 'models/enhanced_model/best_model.pth'
MODEL_NAME = 'HooshvareLab/bert-fa-base-uncased'
TEST_DATA_PATH = 'data/processed_data/test_enhanced.csv'


@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EnhancedPersianSentimentModel(
        MODEL_NAME, num_classes=3, dropout_rate=0.3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer, device


preprocessor = EnhancedPersianPreprocessor()
model, tokenizer, device = load_model_and_tokenizer()
class_names = ['منفی', 'خنثی', 'مثبت']

st.set_page_config(page_title="تحلیل احساسات فارسی", page_icon="💬", layout="centered")

# CSS for right-to-left alignment and optimized width
st.markdown("""
<style>
    .main .block-container {
        max-width: 800px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    body{
        direction: rtl;
    }
    .main-header {
        text-align: right;
        direction: rtl;
    }
    .main-content {
        text-align: right;
        direction: rtl;
    }
    .stTextArea textarea {
        text-align: right;
        direction: rtl;
    }
    .stButton button {
        text-align: center;
    }
    .history-section {
        text-align: right;
        direction: rtl;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💬 تحلیل احساسات متن فارسی با مدل تقویت شده</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-content">یک متن فارسی وارد کنید تا احساس آن (مثبت، منفی یا خنثی) را مدل پیش‌بینی کند.</p>', unsafe_allow_html=True)

# --- Prediction History ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

user_text = st.text_area(
    "متن خود را وارد کنید:", 
    height=120,
    placeholder="مثال: این محصول واقعاً عالی است و کیفیت بالایی دارد..."
)
col1, col2 = st.columns([1, 1])
predict_btn = col1.button("🔍 پیش‌بینی احساس", use_container_width=True)
clear_btn = col2.button("🗑️ پاک‌کردن تاریخچه", use_container_width=True)

if clear_btn:
    st.session_state['history'] = []
    st.info("تاریخچه پاک شد.")

if predict_btn:
    if not user_text.strip():
        st.warning("لطفاً یک متن وارد کنید.")
    else:
        try:
            with st.spinner("در حال پیش‌پردازش و پیش‌بینی..."):
                processed = preprocessor.preprocess_text(user_text)
                encoding = tokenizer(
                    processed,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                with torch.no_grad():
                    logits = model(input_ids=input_ids,
                                   attention_mask=attention_mask)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred = int(np.argmax(probs))
                st.success(f"نتیجه: **{class_names[pred]}**")
                st.info(f"متن پردازش‌شده: {processed}")
                st.subheader("احتمال هر کلاس:")
                
                # Create proper dataframe for chart with Persian labels
                chart_data = pd.DataFrame({
                    'احساس': class_names,
                    'احتمال': probs
                })
                st.bar_chart(chart_data.set_index('احساس'), use_container_width=True)
                # Add to history
                st.session_state['history'].append({
                    'متن': user_text,
                    'پیش‌بینی': class_names[pred],
                    'احتمال منفی': f"{probs[0]*100:.1f}%",
                    'احتمال خنثی': f"{probs[1]*100:.1f}%",
                    'احتمال مثبت': f"{probs[2]*100:.1f}%"
                })
        except Exception as e:
            st.error(f"خطا در پیش‌بینی: {e}")

# --- Prediction History Table ---
if st.session_state['history']:
    st.markdown("---")
    st.markdown('<h3 class="history-section">تاریخچه پیش‌بینی‌ها (این جلسه):</h3>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(st.session_state['history']))

st.markdown("---")

# --- Academic Citation ---
st.markdown('<h3 class="main-content">📚 مراجع علمی</h3>', unsafe_allow_html=True)
with st.expander("مشاهده مراجع"):
    st.markdown("""
    <div dir="ltr" style="text-align: left; line-height: 1.6; font-size: 14px;">
        <p style="margin: 0 0 8px 0; font-weight: 600;">
            <strong>Hosseini, P., Ramaki, A. A., Maleki, H., Anvari, M., & Mirroshandel, S. A. (2018).</strong>
        </p>
        <p style="margin: 0 0 8px 0;">
            SentiPers: A sentiment analysis corpus for Persian.
        </p>
        <p style="margin: 0; font-style: italic; color: #6c757d;">
            <em>arXiv preprint arXiv:1801.07737</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- Developer Credits ---
st.markdown("---")
st.markdown("""
<div dir="rtl" style="text-align: center; padding: 16px 0; margin: 24px 0;">
    <p style="margin: 0; font-size: 14px; color: #6c757d; line-height: 1.5; font-weight: 400;">
        <span style="color: #495057; font-weight: 500;">توسعه‌دهندگان:</span>
        <span style="color: #495057; font-weight: 400;"> سینا مرادیان، علی شجاعیان، سعید مرادعلیان</span>
    </p>
    <p style="margin: 4px 0 0 0; font-size: 12px; color: #adb5bd; line-height: 1.4;">
        با کمک ابزارهای هوش مصنوعی • تابستان ۱۴۰۴
    </p>
</div>
""", unsafe_allow_html=True)
