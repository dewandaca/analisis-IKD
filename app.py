
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
    elif page == "Import & Prediksi":
        st.title("Import & Prediksi Komentar")
        
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




