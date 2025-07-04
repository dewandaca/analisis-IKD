
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from auth import Authenticator
import collections
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
import io
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np

# Download stopwords (kalau belum)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

st.set_page_config(page_title="Analisis Sentimen Aplikasi", layout="wide")

# Inisialisasi autentikasi
auth = Authenticator()

# Fungsi preprocessing dari data_clean.py
def initialize_preprocessing():
    # Stopwords Bahasa Indonesia
    stop_words = set(stopwords.words('indonesian'))

    # Custom Stopwords
    custom_stopwords = {
    "aplikasi","apk","app","banget", "aja", "nih", "dong", "ya", "deh", "sih",
    "udah", "cuma", "lagi", "baru", "malah", "gitu", "kok", "loh", "yaudah",
    "terus", "kayak", "mau", "semoga","yg","yang","nya","dukcapil","kk","ktp",
    "pemerintah","tidak","tdk","gk","gak",
    }
    stop_words = stop_words.union(custom_stopwords)
    
    # Inisialisasi stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    # Load slang dictionary
    slang_dict = {}
    try:
        with open('slang.txt', 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':')
                    slang_dict[key.strip()] = value.strip()
    except Exception as e:
        st.warning(f"Gagal memuat kamus slang: {e}")
    
    return stop_words, stemmer, slang_dict

# Fungsi preprocessing
def clean_text(text, stop_words):
    text = re.sub(r"http\S+|www\S+", "", text)  # hapus URL
    text = re.sub(r"\@[\w]*", "", text)  # hapus mention
    text = re.sub(r"[^\w\s]", "", text)  # hapus tanda baca
    text = re.sub(r"\d+", "", text)  # hapus angka
    text = text.strip()  # hapus spasi di awal/akhir
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]  # hapus stopwords
    return " ".join(tokens)

def clearEmoji(content):
    return content.encode('ascii', 'ignore').decode('ascii')

def replaceTOM(content):
    pola = re.compile(r'(.)\1{2,}', re.DOTALL)
    return pola.sub(r'\1', content)

def casefoldingText(content):
    return content.lower()

def hapus_stopword(text):
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

def replace_slang(text, slang_dict):
    words = text.split()
    words = [slang_dict.get(word, word) for word in words]
    return " ".join(words)

def stem_text(text, stemmer):
    return stemmer.stem(text)

# Fungsi untuk memproses dataframe
def process_dataframe(df, stop_words, stemmer, slang_dict):
    # Pastikan ada kolom 'content'
    if 'content' not in df.columns:
        st.error("Data harus memiliki kolom 'content' yang berisi teks komentar")
        return None
    
    # Hapus baris dengan komentar kosong
    df = df.dropna(subset=['content'])
    df = df[df['content'].str.strip() != '']
    
    # Batasi jumlah komentar menjadi 100
    if len(df) > 100:
        df = df.head(100)
        st.warning("Data dibatasi hingga 100 komentar pertama")
    
    # Terapkan preprocessing
    df['CaseFolding'] = df['content'].apply(casefoldingText)
    df['Cleaning'] = df['CaseFolding'].apply(lambda x: clean_text(x, stop_words))
    
    # Hapus baris dengan hasil cleaning kosong
    df = df[df['Cleaning'].str.strip() != '']
    
    df['HapusEmoji'] = df['Cleaning'].apply(clearEmoji)
    df['3/Lebih'] = df['HapusEmoji'].apply(replaceTOM)
    df['normalisasi'] = df['3/Lebih'].apply(lambda x: replace_slang(str(x).lower(), slang_dict))
    df['normalisasi_stopword'] = df['normalisasi'].apply(hapus_stopword)
    df['Stemming'] = df['normalisasi_stopword'].apply(lambda x: stem_text(x, stemmer))
    
    return df

# Load model IndoBERT
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("indobert_model_IKD2")
    tokenizer = BertTokenizer.from_pretrained("indobert_model_IKD2")
    return model, tokenizer

# Fungsi untuk memuat model alternatif (indobert_IKD_final)
@st.cache_resource
def load_alternative_model():
    model = BertForSequenceClassification.from_pretrained("indobert_IKD_final")
    tokenizer = BertTokenizer.from_pretrained("indobert_IKD_final")
    return model, tokenizer

# Fungsi untuk membandingkan hasil prediksi dari dua model
def compare_predictions(text, model1, tokenizer1, model2, tokenizer2):
    # Prediksi dengan model pertama (indobert_model_IKD2)
    label1, confidence1 = predict_sentiment(text, model1, tokenizer1)
    
    # Prediksi dengan model kedua (indobert_IKD_final)
    label2, confidence2 = predict_sentiment(text, model2, tokenizer2)
    
    return {
        "model1": {
            "name": "indobert_model_IKD2",
            "label": label1,
            "confidence": confidence1
        },
        "model2": {
            "name": "indobert_IKD_final",
            "label": label2,
            "confidence": confidence2
        },
        "match": label1 == label2
    }

# Fungsi prediksi sentimen
def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
    label_idx = torch.argmax(probs).item()
    
    # Mapping label index ke teks
    label_map = {0: "negatif", 1: "netral", 2: "positif"}
    return label_map[label_idx], probs[label_idx].item()

# Fungsi untuk memprediksi batch komentar
def predict_batch(df, model, tokenizer):
    results = []
    
    for idx, row in df.iterrows():
        # Gunakan kolom 'content' untuk prediksi, bukan 'Stemming'
        text = row['content']
        label, confidence = predict_sentiment(text, model, tokenizer)
        results.append({
            'label': label,
            'confidence': confidence
        })
    
    # Tambahkan hasil prediksi ke dataframe
    result_df = df.copy()
    result_df['predicted_label'] = [r['label'] for r in results]
    result_df['confidence'] = [r['confidence'] for r in results]
    
    return result_df

# Fungsi untuk membersihkan dataframe
def clean_dataframe(df):
    # Hapus baris dengan komentar kosong
    df = df.dropna(subset=['content'])
    df = df[df['content'].str.strip() != '']
    
    # Hapus kolom yang seluruhnya kosong
    df = df.dropna(axis=1, how='all')
    
    # Hapus kolom yang lebih dari 50% kosong
    threshold = len(df) * 0.5
    df = df.dropna(axis=1, thresh=threshold)
    
    return df

# Sidebar untuk navigasi dan login/logout
with st.sidebar:
    st.title("Menu")
    
    if auth.is_authenticated():
        user = auth.get_user()
        st.success(f"Selamat datang, {user['name'] or user['username']}!")
        
        # Menu navigasi
        page = st.radio(
            "Pilih Halaman:",
            ["Dashboard", "Analisis Sentimen", "Prediksi Sentimen", "Lexicon & WordCloud", "Impor & Prediksi", "Perbandingan Model"]
        )
        
        if st.button("Logout"):
            auth.logout()
            st.rerun()
    else:
        st.info("Silakan login untuk mengakses aplikasi")
        tab1, tab2 = st.tabs(["Login", "Daftar"])
        
        with tab1:
            auth.login_form()
        
        with tab2:
            auth.register_form()
        
        # Set halaman default untuk pengguna yang belum login
        page = "Login"

# Hanya tampilkan konten jika pengguna sudah login
if auth.is_authenticated():
    if page == "Dashboard":
        st.title("📱 Analisis Sentimen Aplikasi Identitas Kependudukan")
        
        # Tampilkan dashboard
        st.header("Dashboard")
        
        # Load data
        try:
            df_hasil = pd.read_csv("hasil_labeling_stemming(deploy).csv")
            
            # Tampilkan statistik
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_reviews = len(df_hasil)
                st.metric("Total Ulasan", total_reviews)
            
            with col2:
                positif = len(df_hasil[df_hasil['label'] == 'positif'])
                st.metric("Sentimen Positif", positif, f"{positif/total_reviews:.1%}")
            
            with col3:
                netral = len(df_hasil[df_hasil['label'] == 'netral'])
                st.metric("Sentimen Netral", netral, f"{netral/total_reviews:.1%}")
            
            with col4:
                negatif = len(df_hasil[df_hasil['label'] == 'negatif'])
                st.metric("Sentimen Negatif", negatif, f"{negatif/total_reviews:.1%}")
            
            # Visualisasi distribusi sentimen
            st.subheader("Distribusi Sentimen")
            sentimen_counts = df_hasil['label'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sentimen_counts.plot(kind='bar', color=['green', 'red', 'gray'], ax=ax)
            ax.set_xlabel('Sentimen')
            ax.set_ylabel('Jumlah Ulasan')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    elif page == "Analisis Sentimen":
        st.title("Analisis Sentimen")
        
        # Load data
        try:
            df_hasil = pd.read_csv("hasil_labeling_stemming(deploy).csv")
            
            # Filter dan tampilkan data
            st.subheader("Data Ulasan")
            
            # Filter berdasarkan sentimen
            sentimen_filter = st.multiselect(
                "Filter berdasarkan sentimen:",
                options=["positif", "netral", "negatif"],
                default=["positif", "netral", "negatif"]
            )
            
            filtered_df = df_hasil[df_hasil['label'].isin(sentimen_filter)]
            
            # Tampilkan semua kolom data
            st.dataframe(
                filtered_df,
                use_container_width=True
            )
            
            # Word Cloud
            st.subheader("Word Cloud")
            
            sentimen_wc = st.selectbox(
                "Pilih sentimen untuk word cloud:",
                options=["Semua", "positif", "netral", "negatif"]
            )
            
            if sentimen_wc == "Semua":
                text = " ".join(df_hasil['Stemming'].dropna())
            else:
                text = " ".join(df_hasil[df_hasil['label'] == sentimen_wc]['Stemming'].dropna())
            
            if text:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.info("Tidak ada data untuk ditampilkan")
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    elif page == "Prediksi Sentimen":
        st.title("Prediksi Sentimen")
        
        # Load model IndoBERT
        model, tokenizer = load_model()

        # Tambahkan halaman perbandingan model
    elif page == "Perbandingan Model":
        st.title("Perbandingan Model Analisis Sentimen")
        
        # Load kedua model
        model1, tokenizer1 = load_model()
        model2, tokenizer2 = load_alternative_model()
        
        st.write("""
        ### Informasi Model
        
        Aplikasi ini menggunakan dua model IndoBERT yang telah di-fine-tune untuk analisis sentimen:
        
        1. **indobert_model_IKD2** - Model default yang digunakan dalam aplikasi
        2. **indobert_IKD_final** - Model alternatif untuk perbandingan
        
        Kedua model menggunakan arsitektur BERT untuk klasifikasi teks dengan 3 kelas sentimen: positif, netral, dan negatif.
        """)
        
        # Tambahkan tab untuk input teks dan input file
        input_tab1, input_tab2 = st.tabs(["Input Teks", "Input File"])
        
        with input_tab1:
            # Form untuk input teks
            st.subheader("Uji Perbandingan Model")
            user_input = st.text_area("Masukkan teks ulasan untuk dianalisis:", height=150)
            
            if st.button("Bandingkan Hasil Prediksi", key="compare_text"):
                if user_input:
                    with st.spinner("Memproses..."):
                        # Bandingkan hasil prediksi dari kedua model
                        comparison = compare_predictions(user_input, model1, tokenizer1, model2, tokenizer2)
                        
                        # Tampilkan hasil perbandingan
                        st.subheader("Hasil Perbandingan")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Model: {comparison['model1']['name']}**")
                            sentiment1 = comparison['model1']['label']
                            confidence1 = comparison['model1']['confidence']
                            
                            if sentiment1 == "positif":
                                st.success(f"Sentimen: {sentiment1}")
                            elif sentiment1 == "netral":
                                st.info(f"Sentimen: {sentiment1}")
                            else:
                                st.error(f"Sentimen: {sentiment1}")
                            
                            st.write(f"Confidence: {confidence1:.4f}")
                        
                        with col2:
                            st.write(f"**Model: {comparison['model2']['name']}**")
                            sentiment2 = comparison['model2']['label']
                            confidence2 = comparison['model2']['confidence']
                            
                            if sentiment2 == "positif":
                                st.success(f"Sentimen: {sentiment2}")
                            elif sentiment2 == "netral":
                                st.info(f"Sentimen: {sentiment2}")
                            else:
                                st.error(f"Sentimen: {sentiment2}")
                            
                            st.write(f"Confidence: {confidence2:.4f}")
                        
                        # Tampilkan apakah kedua model memberikan hasil yang sama
                        if comparison['match']:
                            st.success("✅ Kedua model memberikan hasil prediksi yang sama.")
                        else:
                            st.warning("⚠️ Kedua model memberikan hasil prediksi yang berbeda.")
                        
                        # Visualisasi confidence scores
                        st.subheader("Perbandingan Confidence Score")
                        
                        confidence_data = {
                            'Model': [comparison['model1']['name'], comparison['model2']['name']],
                            'Confidence': [comparison['model1']['confidence'], comparison['model2']['confidence']]
                        }
                        
                        confidence_df = pd.DataFrame(confidence_data)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        bars = ax.bar(confidence_df['Model'], confidence_df['Confidence'], color=['skyblue', 'lightgreen'])
                        ax.set_ylim(0, 1)
                        ax.set_ylabel('Confidence Score')
                        ax.set_title('Perbandingan Confidence Score Antar Model')
                        
                        # Tambahkan label nilai di atas bar
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                    f'{height:.4f}', ha='center', va='bottom')
                        
                        st.pyplot(fig)
                else:
                    st.warning("Silakan masukkan teks terlebih dahulu")
        
        with input_tab2:
            st.subheader("Impor File untuk Perbandingan")
            
            # Opsi upload file
            upload_option = st.radio(
                "Pilih metode input:",
                ["Unggah File CSV", "Unggah File Excel"]
            )
            
            if upload_option == "Unggah File CSV":
                uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
                
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.success(f"File berhasil diunggah: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error saat membaca file CSV: {e}")
                        df = None
            else:
                uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx", "xls"])
                
                if uploaded_file is not None:
                    try:
                        df = pd.read_excel(uploaded_file)
                        st.success(f"File berhasil diunggah: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error saat membaca file Excel: {e}")
                        df = None
            
            # Jika file berhasil diunggah
            if 'df' in locals() and df is not None:
                # Tampilkan preview data
                st.subheader("Preview Data")
                st.dataframe(df.head())
                
                # Pilih kolom yang berisi teks komentar
                if len(df.columns) > 0:
                    # Coba temukan kolom yang mungkin berisi teks ulasan
                    text_column_candidates = [col for col in df.columns if col.lower() in ['content', 'text', 'review', 'komentar', 'ulasan', 'comment']]
                    
                    # Jika ada kandidat, gunakan sebagai default, jika tidak gunakan kolom pertama
                    default_column = text_column_candidates[0] if text_column_candidates else df.columns[0]
                    
                    # Tambahkan petunjuk yang jelas untuk memilih kolom teks
                    st.info("Pilih kolom yang berisi teks ulasan/komentar (bukan ID atau metadata lainnya)")
                    
                    text_column = st.selectbox(
                        "Pilih kolom yang berisi teks komentar:",
                        options=df.columns,
                        index=df.columns.get_loc(default_column) if default_column in df.columns else 0
                    )
                    
                    # Tampilkan contoh teks dari kolom yang dipilih
                    if not df[text_column].empty:
                        st.write("**Contoh teks dari kolom yang dipilih:**")
                        st.write(df[text_column].iloc[0])
                    
                    # Batasi jumlah data untuk prediksi
                    max_rows = min(len(df), 100)
                    num_rows = st.slider("Jumlah baris untuk diproses:", 1, max_rows, min(20, max_rows))
                    
                    if st.button("Bandingkan Model", key="compare_file"):
                        if text_column:
                            with st.spinner(f"Memproses {num_rows} baris data..."):
                                # Ambil subset data
                                subset_df = df.head(num_rows).copy()
                                
                                # Pastikan kolom teks tidak kosong
                                subset_df = subset_df.dropna(subset=[text_column])
                                subset_df = subset_df[subset_df[text_column].str.strip() != '']
                                
                                if len(subset_df) > 0:
                                    # Inisialisasi hasil
                                    results = []
                                    
                                    # Proses setiap baris
                                    for idx, row in subset_df.iterrows():
                                        text = str(row[text_column])
                                        # Tambahkan validasi untuk memastikan teks tidak hanya berisi angka (seperti ID)
                                        if not text.isdigit() and len(text.strip()) > 5:  # Minimal 5 karakter
                                            comparison = compare_predictions(text, model1, tokenizer1, model2, tokenizer2)
                                            
                                            results.append({
                                                'text': text,
                                                'model1_label': comparison['model1']['label'],
                                                'model1_confidence': comparison['model1']['confidence'],
                                                'model2_label': comparison['model2']['label'],
                                                'model2_confidence': comparison['model2']['confidence'],
                                                'match': comparison['match']
                                            })
                                        else:
                                            st.warning(f"Baris dengan teks '{text}' dilewati karena terlalu pendek atau hanya berisi angka.")
                                    
                                    if results:
                                        # Buat DataFrame hasil
                                        results_df = pd.DataFrame(results)
                                        
                                        # Tampilkan hasil
                                        st.subheader("Hasil Perbandingan")
                                        st.dataframe(results_df)
                                        
                                        # Hitung statistik
                                        match_percentage = (results_df['match'].sum() / len(results_df)) * 100
                                        
                                        st.write(f"**Persentase Kecocokan:** {match_percentage:.2f}%")
                                        
                                        # Visualisasi distribusi label
                                        st.subheader("Distribusi Label")
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write(f"**Model: indobert_model_IKD2**")
                                            model1_counts = results_df['model1_label'].value_counts()
                                            
                                            fig1, ax1 = plt.subplots(figsize=(8, 8))
                                            colors = ['#ff9999', '#66b3ff', '#99ff99']
                                            ax1.pie(model1_counts, labels=model1_counts.index, autopct='%1.1f%%', colors=colors)
                                            ax1.set_title('Distribusi Label - Model 1')
                                            st.pyplot(fig1)
                                        
                                        with col2:
                                            st.write(f"**Model: indobert_IKD_final**")
                                            model2_counts = results_df['model2_label'].value_counts()
                                            
                                            fig2, ax2 = plt.subplots(figsize=(8, 8))
                                            ax2.pie(model2_counts, labels=model2_counts.index, autopct='%1.1f%%', colors=colors)
                                            ax2.set_title('Distribusi Label - Model 2')
                                            st.pyplot(fig2)
                                    else:
                                        st.error("Tidak ada teks valid yang dapat diproses. Pastikan kolom yang dipilih berisi teks ulasan, bukan ID atau metadata.")
                                else:
                                    st.warning("Tidak ada data yang valid untuk diproses setelah menghapus baris kosong.")
                        else:
                            st.warning("Silakan pilih kolom yang berisi teks komentar.")
                else:
                    st.warning("File yang diunggah tidak memiliki kolom.")

    elif page == "Lexicon & WordCloud":
        st.title("Lexicon & WordCloud")
        
        # Inisialisasi preprocessing
        stop_words, stemmer, slang_dict = initialize_preprocessing()
        
        # Load data
        try:
            df_hasil = pd.read_csv("hasil_labeling_stemming(deploy).csv")
            
            # Pilihan sentimen
            sentimen_option = st.selectbox(
                "Pilih sentimen:",
                options=["Semua", "positif", "netral", "negatif"]
            )
            
            # Filter data berdasarkan sentimen
            if sentimen_option == "Semua":
                filtered_df = df_hasil
            else:
                filtered_df = df_hasil[df_hasil['label'] == sentimen_option]
            
            # Tampilkan WordCloud jika ada data
            if filtered_df is not None and not filtered_df.empty:
                # Buat container untuk wordcloud
                wordcloud_container = st.container()
                
                # Tampilkan WordCloud
                st.subheader("Word Cloud")
                
                with wordcloud_container:
                    if 'Stemming' in filtered_df.columns:
                        text = " ".join(filtered_df['Stemming'].dropna().astype(str))
                        
                        if text:
                            try:
                                # Buat wordcloud dengan parameter yang lebih stabil
                                wordcloud = WordCloud(
                                    width=800, 
                                    height=400, 
                                    background_color='white',
                                    max_words=100,
                                    contour_width=3,
                                    contour_color='steelblue',
                                    collocations=False  # Hindari duplikasi kata
                                ).generate(text)
                                
                                # Tampilkan wordcloud
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error saat membuat wordcloud: {e}")
                        else:
                            st.info(f"Tidak ada data untuk sentimen '{sentimen_option}' yang dapat ditampilkan")
                
                # Tampilkan Lexicon (frekuensi kata)
                with st.expander("Lihat Lexicon (Frekuensi Kata)", expanded=True):
                    st.subheader("Lexicon (Frekuensi Kata)")
                    
                    if 'Stemming' in filtered_df.columns:
                        # Gabungkan semua kata
                        all_words = " ".join(filtered_df['Stemming'].dropna().astype(str)).split()
                        
                        # Hitung frekuensi kata
                        word_counts = Counter(all_words)
                        
                        # Konversi ke DataFrame
                        lexicon_df = pd.DataFrame(word_counts.most_common(50), columns=['Kata', 'Frekuensi'])
                        
                        # Tampilkan tabel
                        st.dataframe(lexicon_df, use_container_width=True)
                        
                        # Visualisasi frekuensi kata (top 20)
                        if not lexicon_df.empty:
                            # Ambil top words sesuai jumlah yang tersedia (maksimal 20)
                            top_words = lexicon_df.head(min(20, len(lexicon_df)))
                            n_words = len(top_words)
                            
                            fig, ax = plt.subplots(figsize=(12, 8))
                            bars = ax.barh(top_words['Kata'][::-1], top_words['Frekuensi'][::-1], color='skyblue')
                            ax.set_xlabel('Frekuensi')
                            ax.set_ylabel('Kata')
                            ax.set_title(f'Top {n_words} Kata - {sentimen_option}')
                            
                            # Tambahkan label frekuensi
                            for i, bar in enumerate(bars):
                                ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                                        str(top_words['Frekuensi'].iloc[n_words-1-i]),
                                        va='center')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.info("Tidak ada data frekuensi kata untuk ditampilkan")
                    else:
                        st.info("Tidak ada data lexicon untuk ditampilkan")
                
                # Tambahkan fitur download data
                st.subheader("Download Data")
                
                # Fungsi untuk mengkonversi dataframe ke CSV
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')
                
                # Tombol download
                st.download_button(
                    label="Download Data Lexicon",
                    data=convert_df_to_csv(lexicon_df),
                    file_name=f"lexicon_{sentimen_option}.csv",
                    mime="text/csv",
                    help="Klik untuk mengunduh data lexicon dalam format CSV"
                )
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    # Pada bagian "Impor & Prediksi", temukan kode yang menangani wordcloud
    elif page == "Impor & Prediksi":
        st.title("Impor & Prediksi Komentar")
        
        # Inisialisasi preprocessing
        stop_words, stemmer, slang_dict = initialize_preprocessing()
        
        # Load model
        model, tokenizer = load_model()
        
        # Upload file CSV
        uploaded_file = st.file_uploader("Unggah file CSV komentar", type=["csv"])
        
        # Contoh format
        st.info("""
        Format CSV yang diharapkan:
        - Harus memiliki kolom 'content' yang berisi teks komentar
        - Kolom lain opsional (misalnya: 'userName', 'at', dll)
        
        Contoh:
        ```
        content,userName,at
        "Aplikasi ini bagus sekali",User1,2023-01-01
        "Saya tidak suka dengan aplikasi ini",User2,2023-01-02
        ```
        """)
        
        # Atau input manual
        st.subheader("Atau masukkan komentar secara manual")
        
        manual_input = st.text_area(
            "Masukkan komentar (satu komentar per baris, maksimal 100 komentar):",
            height=200
        )
        
        # Tombol proses
        col1, col2 = st.columns(2)
        with col1:
            process_upload = st.button("Proses File CSV", disabled=uploaded_file is None)
        with col2:
            process_manual = st.button("Proses Komentar Manual", disabled=not manual_input)
        
        # Variabel untuk menyimpan dataframe hasil
        result_df = None
        
        # Proses data dari file CSV
        if process_upload and uploaded_file is not None:
            try:
                # Baca file CSV
                df = pd.read_csv(uploaded_file)
                
                # Pastikan ada kolom 'content'
                if 'content' not in df.columns:
                    st.error("Data harus memiliki kolom 'content' yang berisi teks komentar")
                else:
                    # Bersihkan dataframe
                    df = clean_dataframe(df)
                    
                    # Batasi jumlah komentar menjadi 100
                    if len(df) > 100:
                        df = df.head(100)
                        st.warning("Data dibatasi hingga 100 komentar pertama")
                    
                    if not df.empty:
                        # Proses dataframe dengan preprocessing
                        processed_df = process_dataframe(df, stop_words, stemmer, slang_dict)
                        
                        if processed_df is not None:
                            # Prediksi sentimen
                            with st.spinner("Memproses data..."):
                                result_df = predict_batch(processed_df, model, tokenizer)
                                
                                # Simpan ke session state untuk digunakan nanti
                                st.session_state.result_df = result_df
                                
                                # Tampilkan hasil
                                st.success(f"Berhasil memproses {len(result_df)} komentar")
            except Exception as e:
                st.error(f"Error saat memproses file: {e}")
        
        # Proses data dari input manual
        elif process_manual and manual_input:
            try:
                # Buat dataframe dari input manual
                lines = [line.strip() for line in manual_input.split('\n') if line.strip()]
                
                # Batasi jumlah komentar menjadi 100
                if len(lines) > 100:
                    lines = lines[:100]
                    st.warning("Data dibatasi hingga 100 komentar pertama")
                
                # Buat dataframe
                df = pd.DataFrame({"content": lines})
                
                # Proses dataframe dengan preprocessing
                processed_df = process_dataframe(df, stop_words, stemmer, slang_dict)
                
                if processed_df is not None:
                    # Prediksi sentimen
                    with st.spinner("Memproses data..."):
                        result_df = predict_batch(processed_df, model, tokenizer)
                        
                        # Simpan ke session state untuk digunakan nanti
                        st.session_state.result_df = result_df
                        
                        # Tampilkan hasil
                        st.success(f"Berhasil memproses {len(result_df)} komentar")
            except Exception as e:
                st.error(f"Error saat memproses input manual: {e}")
        
        # Gunakan result_df dari session state jika ada
        if 'result_df' in st.session_state and st.session_state.result_df is not None:
            result_df = st.session_state.result_df
        
        # Tampilkan hasil analisis jika ada data
        if result_df is not None:
            # Tampilkan statistik
            st.subheader("Hasil Analisis Sentimen")
            
            # Hitung statistik
            sentimen_counts = result_df['predicted_label'].value_counts()
            total = len(result_df)
            
            # Pastikan semua label ada
            for label in ['positif', 'netral', 'negatif']:
                if label not in sentimen_counts:
                    sentimen_counts[label] = 0
            
            # Tampilkan metrik
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positif", sentimen_counts.get('positif', 0), 
                         f"{sentimen_counts.get('positif', 0)/total:.1%}")
            with col2:
                st.metric("Netral", sentimen_counts.get('netral', 0), 
                         f"{sentimen_counts.get('netral', 0)/total:.1%}")
            with col3:
                st.metric("Negatif", sentimen_counts.get('negatif', 0), 
                         f"{sentimen_counts.get('negatif', 0)/total:.1%}")
            
            # Visualisasi distribusi sentimen
            st.subheader("Distribusi Sentimen")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = {'positif': 'green', 'netral': 'gray', 'negatif': 'red'}
            sentimen_counts.plot(kind='bar', color=[colors.get(x, 'blue') for x in sentimen_counts.index], ax=ax)
            ax.set_xlabel('Sentimen')
            ax.set_ylabel('Jumlah Komentar')
            ax.set_title('Distribusi Sentimen')
            st.pyplot(fig)
            
            # Tampilkan data hasil prediksi
            st.subheader("Data Hasil Prediksi")
            
            # Tampilkan dataframe
            st.dataframe(result_df, use_container_width=True)
            
            # Tambahkan fitur download data
            st.subheader("Download Data Hasil Prediksi")
            
            # Pilihan format download
            download_format = st.radio(
                "Pilih format download:",
                ["CSV (semua kolom)", "CSV (kolom utama saja)"]
            )
            
            # Fungsi untuk mengkonversi dataframe ke CSV
            def convert_df_to_csv(df, include_all_columns=True):
                # Jika tidak semua kolom, hanya sertakan kolom utama
                if not include_all_columns:
                    df = df[['content', 'predicted_label', 'confidence']]
                
                return df.to_csv(index=False).encode('utf-8')
            
            # Siapkan file untuk didownload
            if download_format == "CSV (semua kolom)":
                csv_data = convert_df_to_csv(result_df, include_all_columns=True)
                file_name = "hasil_prediksi_lengkap.csv"
            else:
                csv_data = convert_df_to_csv(result_df, include_all_columns=False)
                file_name = "hasil_prediksi_ringkas.csv"
            
            # Tombol download
            st.download_button(
                label="Download Data",
                data=csv_data,
                file_name=file_name,
                mime="text/csv",
                help="Klik untuk mengunduh data hasil prediksi dalam format CSV"
            )
            
            # Tambahkan WordCloud
            st.subheader("Word Cloud")
            
            # Pilihan sentimen untuk word cloud
            sentimen_wc = st.selectbox(
                "Pilih sentimen untuk word cloud:",
                options=["Semua", "positif", "netral", "negatif"]
            )
            
            # Filter data berdasarkan sentimen
            if sentimen_wc == "Semua":
                wc_text = " ".join(result_df['Stemming'].dropna())
            else:
                wc_text = " ".join(result_df[result_df['predicted_label'] == sentimen_wc]['Stemming'].dropna())
            
            # Buat word cloud jika ada data
            if wc_text:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wc_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.info("Tidak ada data untuk ditampilkan dalam word cloud")
                
            # Tambahkan Lexicon (frekuensi kata) - Top 20 Word
            with st.expander("Lihat Top 20 Kata", expanded=True):
                st.subheader("Top 20 Kata Berdasarkan Frekuensi")
                
                # Filter data berdasarkan sentimen
                if sentimen_wc == "Semua":
                    filtered_df = result_df
                else:
                    filtered_df = result_df[result_df['predicted_label'] == sentimen_wc]
                
                if 'Stemming' in filtered_df.columns and not filtered_df.empty:
                    # Gabungkan semua kata
                    all_words = " ".join(filtered_df['Stemming'].dropna().astype(str)).split()
                    
                    # Hitung frekuensi kata
                    word_counts = Counter(all_words)
                    
                    # Konversi ke DataFrame
                    lexicon_df = pd.DataFrame(word_counts.most_common(50), columns=['Kata', 'Frekuensi'])
                    
                    # Tampilkan tabel
                    st.dataframe(lexicon_df.head(20), use_container_width=True)
                    
                    # Visualisasi frekuensi kata (top 20)
                    if not lexicon_df.empty:
                        # Ambil top words sesuai jumlah yang tersedia (maksimal 20)
                        top_words = lexicon_df.head(min(20, len(lexicon_df)))
                        n_words = len(top_words)
                        
                        fig, ax = plt.subplots(figsize=(12, 8))
                        bars = ax.barh(top_words['Kata'][::-1], top_words['Frekuensi'][::-1], color='skyblue')
                        ax.set_xlabel('Frekuensi')
                        ax.set_ylabel('Kata')
                        ax.set_title(f'Top {n_words} Kata - {sentimen_wc}')
                        
                        # Tambahkan label frekuensi
                        for i, bar in enumerate(bars):
                            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                                    str(top_words['Frekuensi'].iloc[n_words-1-i]),
                                    va='center')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("Tidak ada data frekuensi kata untuk ditampilkan")
                else:
                    st.info("Tidak ada data untuk ditampilkan")
                
                # Tambahkan tombol download untuk data lexicon
                if 'lexicon_df' in locals() and not lexicon_df.empty:
                    st.download_button(
                        label="Download Data Top 20 Kata",
                        data=convert_df_to_csv(lexicon_df.head(20), include_all_columns=True),
                        file_name=f"top20_kata_{sentimen_wc}.csv",
                        mime="text/csv",
                        help="Klik untuk mengunduh data top 20 kata dalam format CSV"
                    )

else:
    st.title("📱 Analisis Sentimen Aplikasi Identitas Kependudukan")
    st.info("Silakan login untuk mengakses fitur aplikasi")




