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

st.set_page_config(page_title="تحلیل احساسات فارسی", page_icon="💬")
st.title("💬 تحلیل احساسات متن فارسی با مدل حرفه‌ای")
st.write("یک متن فارسی وارد کنید تا احساس آن (مثبت، منفی یا خنثی) را مدل پیش‌بینی کند.")

# --- Prediction History ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

user_text = st.text_area("متن خود را وارد کنید:", height=120)
col1, col2 = st.columns([1, 1])
predict_btn = col1.button("پیش‌بینی احساس")
clear_btn = col2.button("پاک‌کردن تاریخچه")

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
                st.bar_chart({"احتمال": probs}, use_container_width=True)
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
    st.subheader("تاریخچه پیش‌بینی‌ها (این جلسه):")
    st.dataframe(pd.DataFrame(st.session_state['history']))

st.markdown("---")
st.caption("ساخته شده با ❤️ توسط مدل ParsBERT و تیم شما | نسخه پیشرفته GUI")
