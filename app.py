import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import BertTokenizer, BertForSequenceClassification
import torch

st.set_page_config(page_title="Analisis Sentimen Aplikasi", layout="wide")
st.title("📱 Analisis Sentimen Aplikasi Identitas Kependudukan")

# ----------------------
# Load model IndoBERT
# ----------------------
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("indobert_model_IKD2")
    tokenizer = BertTokenizer.from_pretrained("indobert_model_IKD2")
    return model, tokenizer

model, tokenizer = load_model()

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
    label_idx = torch.argmax(probs).item()
    label_map = {0: "negatif", 1: "netral", 2: "positif"}
    return label_map[label_idx], probs.tolist()

# ----------------------
# Sidebar Navigation
# ----------------------
page = st.sidebar.selectbox("Pilih Halaman", ["🔍 Analisis Lexicon", "🧠 Prediksi IndoBERT"])

# ----------------------
# Halaman Analisis Lexicon
# ----------------------
if page == "🔍 Analisis Lexicon":
    st.subheader("📊 Analisis Sentimen dari Lexicon dan IndoBERT")


    df = pd.read_csv('hasil_prediksi_bert.csv')  # Ganti dengan path file

    st.markdown("### Distribusi Sentimen Lexicon")
    label_counts = df['label'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', colors=['green', 'gray', 'red'])
    st.pyplot(fig1)


    selected_sentiment = st.selectbox("Filter berdasarkan sentimen", ["semua", "positif", "netral", "negatif"])
    search_term = st.text_input("Cari komentar...")

    if selected_sentiment != "semua":
        df_filtered = df[df['predicted_label_text'] == selected_sentiment]
    else:
        df_filtered = df

    if search_term:
        df_filtered = df_filtered[df_filtered['content'].str.contains(search_term, case=False)]

    st.write(df_filtered)


    st.markdown("### Contoh Komentar")
    for sentimen in ['positif', 'netral', 'negatif']:
        st.markdown(f"#### {sentimen}")
        sample = df[df['label'] == sentimen]['content'].sample(min(3, len(df[df['label'] == sentimen]))).tolist()
        for i, komentar in enumerate(sample):
            st.write(f"{i+1}. {komentar}")

    st.markdown("### Word Cloud per Sentimen")
    for sentimen, warna in zip(['positif', 'netral', 'negatif'], ['Greens', 'Greys', 'Reds']):
        text = " ".join(df[df['label'] == sentimen]['Stemming'].astype(str).tolist())
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=warna).generate(text)
            st.image(wordcloud.to_array(), caption=f"Word Cloud - {sentimen}")

    st.markdown("### 🔍 Kesimpulan Analisis")
    st.info("""
- Komentar **positif** banyak menyoroti kemudahan dan fitur aplikasi.
- Komentar **netral** umumnya berupa masukan atau observasi.
- Komentar **negatif** dominan pada error sistem, tidak bisa login, dan performa lambat.
    """)

    st.markdown("### ✅ Rekomendasi Perbaikan")
    st.write("""
- Stabilkan sistem login dan backend server.
- Tambahkan panduan pemakaian.
- Tingkatkan kecepatan dan responsivitas aplikasi.
    """)


# ----------------------
# Halaman Prediksi Model IndoBERT
# ----------------------
elif page == "🧠 Prediksi IndoBERT":
    st.subheader("Masukkan Komentar Pengguna")

    user_input = st.text_area("📝 Komentar")

    if st.button("Prediksi Sentimen"):
        if user_input.strip():
            label, probs = predict_sentiment(user_input)
            st.success(f"Sentimen: **{label}**")
            st.write(f"Probabilitas: Positif: {probs[2]:.2f}, Netral: {probs[1]:.2f}, Negatif: {probs[0]:.2f}")

            fig2, ax2 = plt.subplots()
            ax2.bar(['Negatif', 'Netral', 'Positif'], probs, color=['red', 'gray', 'green'])
            ax2.set_ylabel("Probabilitas")
            st.pyplot(fig2)
        else:
            st.warning("Masukkan komentar terlebih dahulu.")



