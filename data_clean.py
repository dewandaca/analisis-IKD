import pandas as pd
import re
import nltk
from nltk.corpus import stopwords


# Download stopwords (kalau belum)
nltk.download('stopwords')

# Stopwords Bahasa Indonesia
stop_words = set(stopwords.words('indonesian'))

# Custom Stopwords
custom_stopwords = {
    "aplikasi","apk","app","banget", "aja", "nih", "dong", "ya", "deh", "sih",
    "udah", "cuma", "lagi", "baru", "malah", "gitu", "kok", "loh", "yaudah",
    "terus", "kayak", "mau", "semoga","yg","yang","nya","dukcapil","kk","ktp","pemerintah",
}
stop_words = stop_words.union(custom_stopwords)



# Fungsi Preprocessing
def clean_text(text):
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



# Terapkan ke DataFrame
new_df_copy = df.copy()
new_df_copy['CaseFolding'] = new_df_copy['content'].apply(casefoldingText)
new_df_copy['Cleaning'] = new_df_copy['CaseFolding'].apply(clean_text)
new_df_copy['HapusEmoji'] = new_df_copy['Cleaning'].apply(clearEmoji)
new_df_copy['3/Lebih'] = new_df_copy['HapusEmoji'].apply(replaceTOM)
# Urutkan berdasarkan waktu
sorted_df = new_df_copy.sort_values(by='at', ascending=False)

# Simpan ke CSV
sorted_df.to_csv("review_IKD_cleaned.csv", index=False)

# normalisasi
slang_dict = {}
with open('slang.txt', 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        if ':' in line:
            key, value = line.strip().split(':')
            slang_dict[key.strip()] = value.strip()

df=pd.read_csv('review_IKD_cleaned.csv')
def replace_slang(text, slang_dict):
    words = text.split()
    words = [slang_dict.get(word, word) for word in words]
    return " ".join(words)

df['normalisasi'] = df['3/Lebih'].apply(lambda x: replace_slang(str(x).lower(), slang_dict))

#Stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stem_text(text):
    return stemmer.stem(text)
# Proses Stemming
df['Stemming'] = df['normalisasi'].apply(stem_text)
# Urutkan berdasarkan waktu
sorted_df = df.sort_values(by='at', ascending=False)