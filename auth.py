import streamlit as st
from user_db import UserDatabase

class Authenticator:
    def __init__(self):
        self.db = UserDatabase()
        # Buat admin default jika belum ada
        self.db.create_admin_if_not_exists()
        
        # Inisialisasi state session
        if "user" not in st.session_state:
            st.session_state.user = None
        if "authentication_status" not in st.session_state:
            st.session_state.authentication_status = None
    
    def login_form(self):
        """Menampilkan form login"""
        with st.form("login_form"):
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if self.authenticate(username, password):
                    st.rerun()
    
    def authenticate(self, username, password):
        """Mengautentikasi pengguna"""
        user = self.db.verify_user(username, password)
        
        if user:
            st.session_state.user = user
            st.session_state.authentication_status = True
            return True
        else:
            st.session_state.authentication_status = False
            st.error("Username atau password salah")
            return False
    
    def logout(self):
        """Logout pengguna"""
        st.session_state.user = None
        st.session_state.authentication_status = None
    
    def register_form(self):
        """Menampilkan form registrasi"""
        with st.form("register_form"):
            st.subheader("Daftar Akun Baru")
            username = st.text_input("Username")
            name = st.text_input("Nama Lengkap")
            password = st.text_input("Password", type="password")
            password_confirm = st.text_input("Konfirmasi Password", type="password")
            
            submitted = st.form_submit_button("Daftar")
            
            if submitted:
                if password != password_confirm:
                    st.error("Password tidak cocok")
                    return False
                
                if self.db.add_user(username, password, name):
                    st.success("Akun berhasil dibuat! Silakan login.")
                    return True
                else:
                    st.error("Username sudah digunakan")
                    return False
    
    def is_authenticated(self):
        """Memeriksa apakah pengguna sudah login"""
        return st.session_state.authentication_status
    
    def get_user(self):
        """Mendapatkan informasi pengguna yang sedang login"""
        return st.session_state.user
    
    def require_login(self):
        """Memastikan pengguna sudah login sebelum mengakses halaman"""
        if not self.is_authenticated():
            st.warning("Anda harus login terlebih dahulu")
            self.login_form()
            return False
        return True