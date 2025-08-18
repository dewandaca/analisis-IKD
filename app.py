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
from google_play_scraper import reviews, Sort
import openpyxl
import zipfile

# Download stopwords (kalau belum)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

st.set_page_config(page_title="Analisis Sentimen Aplikasi", layout="wide")

# Inisialisasi autentikasi
auth = Authenticator()

# Fungsi untuk mengidentifikasi kategori masalah dari teks komentar
def identify_issue_category(text):
    text = text.lower()
    
    # Kata kunci untuk setiap kategori
    login_keywords = ['login', 'masuk', 'daftar', 'registrasi', 'akun', 'password', 'kata sandi', 'username', 'email', 'verifikasi']
    ui_keywords = ['tampilan', 'ui', 'interface', 'desain', 'layout', 'menu', 'tombol', 'button', 'warna', 'font', 'ukuran', 'tata letak']
    bug_keywords = ['bug', 'error', 'crash', 'hang', 'macet', 'berhenti', 'tidak berfungsi', 'tidak bisa', 'gagal', 'force close', 'fc', 'loading']
    
    # Cek kategori berdasarkan kata kunci
    if any(keyword in text for keyword in login_keywords):
        return 'Login'
    elif any(keyword in text for keyword in ui_keywords):
        return 'UI/UX'
    elif any(keyword in text for keyword in bug_keywords):
        return 'Bug'
    else:
        return 'Lainnya'

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
    model = BertForSequenceClassification.from_pretrained("indobert_IKD_final")
    tokenizer = BertTokenizer.from_pretrained("indobert_IKD_final")
    return model, tokenizer

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
            ["Dashboard", "Lexicon & WordCloud", "Import & Prediksi"]
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
            df_hasil = pd.read_csv("hasil_labeling_stemming (4).csv")
            
            # Tambahkan kategori masalah ke dataframe
            if 'kategori_masalah' not in df_hasil.columns:
                df_hasil['kategori_masalah'] = df_hasil['content'].apply(identify_issue_category)
            
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
            
            # Tambahkan analisis distribusi sentimen per kategori masalah
            st.subheader("Distribusi Sentimen per Kategori Masalah")
            
            # Hitung jumlah untuk setiap kombinasi kategori dan sentimen
            kategori_sentimen = pd.crosstab(df_hasil['kategori_masalah'], df_hasil['label'])
            
            # Hitung persentase
            kategori_sentimen_pct = kategori_sentimen.div(kategori_sentimen.sum(axis=1), axis=0) * 100
            
            # Tampilkan tabel persentase
            st.write("Persentase Sentimen per Kategori:")
            st.dataframe(kategori_sentimen_pct.round(1), use_container_width=True)
            
            # Visualisasi dengan stacked bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            kategori_sentimen_pct.plot(kind='bar', stacked=True, 
                                      color=['red', 'gray', 'green'], 
                                      ax=ax)
            ax.set_xlabel('Kategori Masalah')
            ax.set_ylabel('Persentase (%)')
            ax.set_title('Distribusi Sentimen per Kategori Masalah')
            ax.legend(title='Sentimen')
            
            # Tambahkan label persentase
            for c in ax.containers:
                labels = [f'{v:.1f}%' if v > 0 else '' for v in c.datavalues]
                ax.bar_label(c, labels=labels, label_type='center')
                
            st.pyplot(fig)
            
            # Tambahkan visualisasi jumlah ulasan per kategori
            st.subheader("Jumlah Ulasan per Kategori Masalah")
            kategori_counts = df_hasil['kategori_masalah'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            kategori_counts.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_xlabel('Kategori Masalah')
            ax.set_ylabel('Jumlah Ulasan')
            
            # Tambahkan label jumlah
            for i, v in enumerate(kategori_counts):
                ax.text(i, v + 0.5, str(v), ha='center')
                
            st.pyplot(fig)
            
            # Tampilkan data ulasan per kategori masalah
            st.subheader("Data Ulasan per Kategori Masalah")
            
            # Buat tab untuk setiap kategori masalah
            kategori_tabs = st.tabs(sorted(df_hasil['kategori_masalah'].unique()))
            
            # Isi setiap tab dengan data ulasan untuk kategori tersebut
            for i, kategori in enumerate(sorted(df_hasil['kategori_masalah'].unique())):
                with kategori_tabs[i]:
                    # Filter data berdasarkan kategori
                    kategori_df = df_hasil[df_hasil['kategori_masalah'] == kategori]
                    
                    # Tampilkan ringkasan sentimen untuk kategori ini
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        positif_kat = len(kategori_df[kategori_df['label'] == 'positif'])
                        st.metric(f"Positif", positif_kat, f"{positif_kat/len(kategori_df):.1%}")
                    with col2:
                        netral_kat = len(kategori_df[kategori_df['label'] == 'netral'])
                        st.metric(f"Netral", netral_kat, f"{netral_kat/len(kategori_df):.1%}")
                    with col3:
                        negatif_kat = len(kategori_df[kategori_df['label'] == 'negatif'])
                        st.metric(f"Negatif", negatif_kat, f"{negatif_kat/len(kategori_df):.1%}")
                    
                    # Tampilkan data ulasan untuk kategori ini
                    st.write(f"Data Ulasan Kategori: {kategori}")
                    
                    # Pilih kolom yang ingin ditampilkan
                    columns_to_display = ['content', 'label', 'confidence']
                    if all(col in kategori_df.columns for col in columns_to_display):
                        st.dataframe(kategori_df[columns_to_display], use_container_width=True)
                    else:
                        st.dataframe(kategori_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
                # Load data
        try:
            df_hasil = pd.read_csv("hasil_labeling_stemming (4).csv")
            
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
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
    

    
    elif page == "Lexicon & WordCloud":
        st.title("Lexicon & WordCloud")
        
        # Inisialisasi preprocessing
        stop_words, stemmer, slang_dict = initialize_preprocessing()
        
        # Load data
        try:
            df_hasil = pd.read_csv("hasil_labeling_stemming (4).csv")
            
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
                
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    # Pada bagian "Impor & Prediksi", temukan kode yang menangani wordcloud
    elif page == "Import & Prediksi":
        st.title("Import & Prediksi Komentar")
        
        # Inisialisasi result_df di awal
        result_df = None
        
        # Inisialisasi preprocessing
        stop_words, stemmer, slang_dict = initialize_preprocessing()
        
        # Load model
        model, tokenizer = load_model()
        
        # Tambahkan tab untuk memilih metode input
        input_tab1, input_tab2, input_tab3 = st.tabs(["Scraping Play Store", "Upload File", "Input Manual"])
        
        # Tab Scraping Play Store
        with input_tab1:
            st.subheader("Scraping Ulasan dari Play Store")
            
            # Input ID aplikasi
            app_id = st.text_input(
                "Masukkan ID Aplikasi Play Store:",
                value=" ",
                help="Contoh: gov.dukcapil.mobile_id untuk aplikasi IKD"
            )
            
            # Input jumlah ulasan
            review_count = st.number_input(
                "Jumlah ulasan yang akan diambil:",
                min_value=10,
                max_value=100,
                value=100,
                step=10,
                help="Maksimal 100 ulasan untuk menghindari waktu proses yang terlalu lama"
            )
            
            # Pilihan sort
            sort_option = st.selectbox(
                "Urutkan berdasarkan:",
                options=["Paling Relevan", "Terbaru"],
                index=0
            )
            
            # Mapping sort option ke Sort enum
            sort_mapping = {
                "Paling Relevan": Sort.MOST_RELEVANT,
                "Terbaru": Sort.NEWEST,
            }
            
            # Tombol untuk scraping
            scrape_button = st.button("Scraping Ulasan")
            
            # Proses scraping jika tombol ditekan
            if scrape_button and app_id:
                try:
                    with st.spinner(f"Mengambil {review_count} ulasan dari Play Store..."):
                        # Lakukan scraping
                        result, _ = reviews(
                            app_id,
                            lang='id',
                            country='id',
                            count=review_count,
                            sort=sort_mapping[sort_option]
                        )
                        
                        # Cek apakah hasil scraping kosong
                        if not result:
                            st.error("Tidak ada ulasan yang ditemukan untuk aplikasi ini.")
                        else:
                            # Konversi ke DataFrame
                            df = pd.DataFrame(result)
                            
                            # Debug: tampilkan kolom yang tersedia
                            st.write("Kolom yang tersedia:", df.columns.tolist())
                            
                            # Pastikan kolom yang diperlukan ada
                            required_columns = ['content', 'userName', 'score', 'at']
                            missing_columns = [col for col in required_columns if col not in df.columns]
                            
                            if missing_columns:
                                st.error(f"Kolom yang diperlukan tidak ditemukan: {', '.join(missing_columns)}")
                                st.write("Struktur data yang diterima:")
                                st.write(df.head(1))
                            else:
                                # Rename kolom untuk menyesuaikan dengan format yang diharapkan
                                df = df.rename(columns={
                                    'content': 'content',
                                    'userName': 'userName',
                                    'at': 'at',
                                    'score': 'rating'
                                })
                                
                                # Bersihkan dataframe
                                df = clean_dataframe(df)
                                
                                if not df.empty:
                                    # Proses dataframe dengan preprocessing
                                    processed_df = process_dataframe(df, stop_words, stemmer, slang_dict)
                                    
                                    if processed_df is not None:
                                        # Prediksi sentimen
                                        result_df = predict_batch(processed_df, model, tokenizer)
                                        
                                        # Simpan ke session state untuk digunakan nanti
                                        st.session_state.result_df = result_df
                                        
                                        # Tampilkan hasil
                                        st.success(f"Berhasil memproses {len(result_df)} ulasan dari Play Store")
                except Exception as e:
                    st.error(f"Error saat scraping Play Store: {str(e)}")
                    st.write("Detail error:", type(e).__name__)
                    
                    # Tambahkan saran untuk mengatasi masalah
                    st.info("""
                    **Saran untuk mengatasi masalah:**
                    1. Pastikan ID aplikasi benar (contoh: gov.dukcapil.mobile_id untuk aplikasi IKD)
                    2. Coba kurangi jumlah ulasan yang diambil
                    3. Coba gunakan metode input lain (Upload File atau Input Manual)
                    """)
        
        # Tab Upload CSV
        with input_tab2:
            # Upload file CSV
            uploaded_file = st.file_uploader("Unggah file CSV atau Excel komentar (maksimal 100 komentar)", type=["csv", "xlsx", "xls"])
            
            # Contoh format
            st.info("""
            Format file yang diharapkan:
            - Harus memiliki kolom 'content' yang berisi teks komentar
            - Kolom lain opsional (misalnya: 'userName', 'at', dll)
            
            Contoh:
            ```
            content,userName,at
            "Aplikasi ini bagus sekali",User1,2023-01-01
            "Saya tidak suka dengan aplikasi ini",User2,2023-01-02
            ```
            """)
            
            # Tombol proses
            process_upload = st.button("Proses File", disabled=uploaded_file is None)
            
            # Proses data dari file CSV atau Excel
            if process_upload and uploaded_file is not None:
                try:
                    # Cek ekstensi file
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    # Baca file sesuai ekstensi
                    if file_extension in ['xlsx', 'xls']:
                        df = pd.read_excel(uploaded_file)
                    else:  # CSV
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
        
        # Tab Input Manual
        with input_tab3:
            # Input manual
            manual_input = st.text_area(
                "Masukkan komentar (satu komentar per baris, maksimal 100 komentar):",
                height=200
            )
            
            # Tombol proses
            process_manual = st.button("Proses Komentar Manual", disabled=not manual_input)
            
            # Proses data dari input manual
            if process_manual and manual_input:
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
            
            # Tampilkan WordCloud dan Lexicon jika ada data stemming
            if 'Stemming' in result_df.columns and not result_df['Stemming'].dropna().empty:
                # WordCloud Section
                st.subheader("Word Cloud")
                
                # Pilihan sentimen untuk wordcloud
                wordcloud_sentimen = st.selectbox(
                    "Pilih sentimen untuk Word Cloud:",
                    options=["Semua", "positif", "netral", "negatif"],
                    key="wordcloud_sentimen"
                )
                
                # Filter data berdasarkan sentimen
                if wordcloud_sentimen == "Semua":
                    filtered_df = result_df
                else:
                    filtered_df = result_df[result_df['predicted_label'] == wordcloud_sentimen]
                
                # Tampilkan WordCloud jika ada data
                if not filtered_df.empty and not filtered_df['Stemming'].dropna().empty:
                    text = " ".join(filtered_df['Stemming'].dropna().astype(str))
                    
                    if text.strip():
                        try:
                            # Buat wordcloud
                            wordcloud = WordCloud(
                                width=800, 
                                height=400, 
                                background_color='white',
                                max_words=100,
                                contour_width=3,
                                contour_color='steelblue',
                                collocations=False
                            ).generate(text)
                            
                            # Tampilkan wordcloud
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title(f'Word Cloud - {wordcloud_sentimen}', fontsize=16, pad=20)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error saat membuat wordcloud: {e}")
                    else:
                        st.info(f"Tidak ada data untuk sentimen '{wordcloud_sentimen}' yang dapat ditampilkan")
                else:
                    st.info(f"Tidak ada data untuk sentimen '{wordcloud_sentimen}' yang dapat ditampilkan")
                
                # Lexicon Section
                st.subheader("Lexicon (Frekuensi Kata)")
                
                # Pilihan sentimen untuk lexicon
                lexicon_sentimen = st.selectbox(
                    "Pilih sentimen untuk Lexicon:",
                    options=["Semua", "positif", "netral", "negatif"],
                    key="lexicon_sentimen"
                )
                
                # Filter data berdasarkan sentimen
                if lexicon_sentimen == "Semua":
                    filtered_df = result_df
                else:
                    filtered_df = result_df[result_df['predicted_label'] == lexicon_sentimen]
                
                # Tampilkan Lexicon jika ada data
                if not filtered_df.empty and not filtered_df['Stemming'].dropna().empty:
                    # Gabungkan semua kata
                    all_words = " ".join(filtered_df['Stemming'].dropna().astype(str)).split()
                    
                    if all_words:
                        # Hitung frekuensi kata
                        word_counts = Counter(all_words)
                        
                        # Konversi ke DataFrame
                        lexicon_df = pd.DataFrame(word_counts.most_common(50), columns=['Kata', 'Frekuensi'])
                        
                        # Tampilkan tabel dalam expander
                        with st.expander("Lihat Tabel Frekuensi Kata", expanded=True):
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
                            ax.set_title(f'Top {n_words} Kata - {lexicon_sentimen}')
                            
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
                        st.info(f"Tidak ada kata yang dapat dianalisis untuk sentimen '{lexicon_sentimen}'")
                else:
                    st.info(f"Tidak ada data untuk sentimen '{lexicon_sentimen}' yang dapat ditampilkan")
            else:
                st.info("Data hasil preprocessing kosong. WordCloud dan Lexicon tidak dapat ditampilkan.")
            
            # Tampilkan data hasil prediksi
            st.subheader("Data Hasil Prediksi")
            
            # Tampilkan dataframe
            st.dataframe(result_df, use_container_width=True)
            
            # Download Section
            st.subheader("Download Data Hasil Prediksi")
            
            # Cek apakah data hasil prediksi kosong
            is_data_empty = result_df is None or result_df.empty
            
            # Cek apakah data Stemming kosong (semua baris kosong setelah preprocessing)
            has_stemming_data = False
            if not is_data_empty and 'Stemming' in result_df.columns:
                has_stemming_data = not result_df['Stemming'].dropna().empty
            
            # Tampilkan peringatan jika data kosong
            if is_data_empty:
                st.warning("Tidak ada data hasil prediksi yang dapat diunduh. Silakan proses data terlebih dahulu.")
            elif not has_stemming_data:
                st.warning("Data hasil preprocessing kosong. Semua teks telah dihapus selama proses pembersihan.")
            
            # Hanya tampilkan opsi download jika ada data
            if not is_data_empty:
                # Fungsi untuk mengkonversi dataframe ke CSV
                def convert_df_to_csv(df, include_all_columns=True):
                    if not include_all_columns:
                        df = df[['content', 'predicted_label', 'confidence']]
                    return df.to_csv(index=False).encode('utf-8')
                
                # Fungsi untuk menyimpan gambar sebagai bytes
                def fig_to_bytes(fig):
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                    buf.seek(0)
                    return buf.getvalue()
                
                # Opsi download dalam kolom
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Pilih data yang ingin diunduh:**")
                    download_csv = st.checkbox("📊 Data CSV Hasil Prediksi", value=True)
                    download_dist_chart = st.checkbox("📈 Grafik Distribusi Sentimen", value=True)
                    
                    # Pilihan format CSV jika opsi CSV dipilih
                    if download_csv:
                        csv_format = st.radio(
                            "Format CSV:",
                            ["Semua kolom", "Kolom utama saja"],
                            help="Kolom utama: content, predicted_label, confidence"
                        )
                        include_all_columns = csv_format == "Semua kolom"
                
                with col2:
                    st.write("**Pilih visualisasi yang ingin diunduh:**")
                    # Hanya aktifkan opsi wordcloud dan lexicon jika ada data stemming
                    download_wordcloud = st.checkbox("☁️ Word Cloud", value=True, disabled=not has_stemming_data)
                    download_lexicon = st.checkbox("📝 Data Top 20 Kata", value=True, disabled=not has_stemming_data)
                    download_lexicon_chart = st.checkbox("📊 Grafik Top 20 Kata", value=True, disabled=not has_stemming_data)
                    
                    # Tampilkan pesan jika opsi dinonaktifkan
                    if not has_stemming_data:
                        st.info("Visualisasi wordcloud dan lexicon tidak tersedia karena data hasil preprocessing kosong.")
                
                # Tombol download utama
                download_button = st.button("🗂️ Download Hasil Analisis (ZIP)", 
                                          type="primary", 
                                          use_container_width=True,
                                          disabled=is_data_empty)
                
                if download_button:
                    if not any([download_csv, download_dist_chart, 
                               (download_wordcloud and has_stemming_data), 
                               (download_lexicon and has_stemming_data), 
                               (download_lexicon_chart and has_stemming_data)]):
                        st.warning("Pilih minimal satu item untuk diunduh!")
                    else:
                        try:
                            # Siapkan ZIP file untuk menampung semua file
                            zip_buffer = io.BytesIO()
                            
                            with st.spinner("Menyiapkan file untuk diunduh..."):
                                # Gunakan zipfile untuk membuat file ZIP
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                    
                                    # 1. Tambahkan CSV jika dipilih
                                    if download_csv:
                                        csv_data = convert_df_to_csv(result_df, include_all_columns)
                                        file_name = "hasil_prediksi_lengkap.csv" if include_all_columns else "hasil_prediksi_ringkas.csv"
                                        zip_file.writestr(file_name, csv_data)
                                    
                                    # 2. Tambahkan grafik distribusi jika dipilih
                                    if download_dist_chart:
                                        dist_fig, dist_ax = plt.subplots(figsize=(10, 6))
                                        colors = {'positif': 'green', 'netral': 'gray', 'negatif': 'red'}
                                        sentimen_counts.plot(kind='bar', color=[colors.get(x, 'blue') for x in sentimen_counts.index], ax=dist_ax)
                                        dist_ax.set_xlabel('Sentimen')
                                        dist_ax.set_ylabel('Jumlah Komentar')
                                        dist_ax.set_title('Distribusi Sentimen')
                                        plt.xticks(rotation=0)
                                        
                                        dist_img = fig_to_bytes(dist_fig)
                                        zip_file.writestr("distribusi_sentimen.png", dist_img)
                                        plt.close(dist_fig)
                                    
                                    # 3. Tambahkan wordcloud jika dipilih dan ada data stemming
                                    if download_wordcloud and has_stemming_data:
                                        # Buat wordcloud untuk setiap sentimen
                                        sentimen_options = ["Semua", "positif", "netral", "negatif"]
                                        
                                        for sentimen in sentimen_options:
                                            # Filter data berdasarkan sentimen
                                            if sentimen == "Semua":
                                                text_data = " ".join(result_df['Stemming'].dropna().astype(str))
                                            else:
                                                filtered_df = result_df[result_df['predicted_label'] == sentimen]
                                                if filtered_df.empty or filtered_df['Stemming'].dropna().empty:
                                                    continue  # Skip jika tidak ada data untuk sentimen ini
                                                text_data = " ".join(filtered_df['Stemming'].dropna().astype(str))
                                            
                                            if text_data.strip():
                                                try:
                                                    wordcloud_obj = WordCloud(
                                                        width=800, 
                                                        height=400, 
                                                        background_color='white',
                                                        max_words=100,
                                                        collocations=False
                                                    ).generate(text_data)
                                                    
                                                    wc_fig, wc_ax = plt.subplots(figsize=(10, 5))
                                                    wc_ax.imshow(wordcloud_obj, interpolation='bilinear')
                                                    wc_ax.axis('off')
                                                    wc_ax.set_title(f'Word Cloud - {sentimen}', fontsize=16, pad=20)
                                                    
                                                    wc_img = fig_to_bytes(wc_fig)
                                                    zip_file.writestr(f"wordcloud_{sentimen.lower()}.png", wc_img)
                                                    plt.close(wc_fig)
                                                except Exception as e:
                                                    st.warning(f"Gagal membuat wordcloud untuk {sentimen}: {e}")
                                    
                                    # 4. Tambahkan data lexicon jika dipilih dan ada data stemming
                                    if download_lexicon and has_stemming_data:
                                        # Buat lexicon untuk setiap sentimen
                                        sentimen_options = ["Semua", "positif", "netral", "negatif"]
                                        
                                        for sentimen in sentimen_options:
                                            # Filter data berdasarkan sentimen
                                            if sentimen == "Semua":
                                                filtered_data = result_df
                                            else:
                                                filtered_data = result_df[result_df['predicted_label'] == sentimen]
                                            
                                            if not filtered_data.empty and not filtered_data['Stemming'].dropna().empty:
                                                # Gabungkan semua kata
                                                all_words = " ".join(filtered_data['Stemming'].dropna().astype(str)).split()
                                                
                                                if all_words:
                                                    # Hitung frekuensi kata
                                                    word_counts = Counter(all_words)
                                                    
                                                    # Konversi ke DataFrame
                                                    lexicon_data = pd.DataFrame(word_counts.most_common(20), columns=['Kata', 'Frekuensi'])
                                                    
                                                    # Simpan sebagai CSV
                                                    lexicon_csv = lexicon_data.to_csv(index=False).encode('utf-8')
                                                    zip_file.writestr(f"top20_kata_{sentimen.lower()}.csv", lexicon_csv)
                                    
                                    # 5. Tambahkan grafik lexicon jika dipilih dan ada data stemming
                                    if download_lexicon_chart and has_stemming_data:
                                        # Buat grafik lexicon untuk setiap sentimen
                                        sentimen_options = ["Semua", "positif", "netral", "negatif"]
                                        
                                        for sentimen in sentimen_options:
                                            # Filter data berdasarkan sentimen
                                            if sentimen == "Semua":
                                                filtered_data = result_df
                                            else:
                                                filtered_data = result_df[result_df['predicted_label'] == sentimen]
                                            
                                            if not filtered_data.empty and not filtered_data['Stemming'].dropna().empty:
                                                # Gabungkan semua kata
                                                all_words = " ".join(filtered_data['Stemming'].dropna().astype(str)).split()
                                                
                                                if all_words:
                                                    # Hitung frekuensi kata
                                                    word_counts = Counter(all_words)
                                                    
                                                    # Konversi ke DataFrame
                                                    top_words = pd.DataFrame(word_counts.most_common(20), columns=['Kata', 'Frekuensi'])
                                                    
                                                    if not top_words.empty:
                                                        # Buat grafik
                                                        lex_fig, lex_ax = plt.subplots(figsize=(12, 8))
                                                        bars = lex_ax.barh(top_words['Kata'][::-1], top_words['Frekuensi'][::-1], color='skyblue')
                                                        lex_ax.set_xlabel('Frekuensi')
                                                        lex_ax.set_ylabel('Kata')
                                                        lex_ax.set_title(f'Top 20 Kata - {sentimen}')
                                                        
                                                        # Tambahkan label frekuensi
                                                        for i, bar in enumerate(bars):
                                                            lex_ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                                                                    str(top_words['Frekuensi'].iloc[min(19, len(top_words)-1)-i]),
                                                                    va='center')
                                                        
                                                        plt.tight_layout()
                                                        lex_img = fig_to_bytes(lex_fig)
                                                        zip_file.writestr(f"grafik_top20_kata_{sentimen.lower()}.png", lex_img)
                                                        plt.close(lex_fig)
                            
                            # Setel pointer ke awal buffer
                            zip_buffer.seek(0)
                            
                            # Buat tombol download untuk file ZIP
                            st.download_button(
                                label="📥 Unduh File ZIP",
                                data=zip_buffer,
                                file_name=f"hasil_analisis_sentimen_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip",
                                help="Klik untuk mengunduh hasil analisis dalam format ZIP",
                                use_container_width=True
                            )
                            
                            st.success("✅ File ZIP berhasil disiapkan! Klik tombol 'Unduh File ZIP' di atas untuk mengunduh.")
                            
                            # Tampilkan ringkasan file yang akan diunduh
                            items = []
                            if download_csv:
                                items.append(f"• Data CSV ({'semua kolom' if include_all_columns else 'kolom utama'})")
                            if download_dist_chart:
                                items.append("• Grafik distribusi sentimen")
                            if download_wordcloud and has_stemming_data:
                                items.append("• Word cloud (semua sentimen)")
                            if download_lexicon and has_stemming_data:
                                items.append("• Data top 20 kata (semua sentimen)")
                            if download_lexicon_chart and has_stemming_data:
                                items.append("• Grafik top 20 kata (semua sentimen)")
                            
                            if items:
                                st.info("**Ringkasan file yang akan diunduh:**\n" + "\n".join(items))
                            
                        except Exception as e:
                            st.error(f"Error saat menyiapkan file download: {e}")
            
            # Buat kolom untuk tombol download visualisasi
            col1, col2 = st.columns(2)
            
            # Simpan gambar distribusi sentimen
            with col1:
                # Buat ulang grafik distribusi sentimen untuk download
                dist_fig, dist_ax = plt.subplots(figsize=(10, 6))
                colors = {'positif': 'green', 'netral': 'gray', 'negatif': 'red'}
                sentimen_counts.plot(kind='bar', color=[colors.get(x, 'blue') for x in sentimen_counts.index], ax=dist_ax)
                dist_ax.set_xlabel('Sentimen')
                dist_ax.set_ylabel('Jumlah Komentar')
                dist_ax.set_title('Distribusi Sentimen')
                
                # Tombol download grafik distribusi

else:
    st.title("📱 Analisis Sentimen Aplikasi Identitas Kependudukan")
    st.info("Silakan login untuk mengakses fitur aplikasi")




