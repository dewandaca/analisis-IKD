import sqlite3
import hashlib
import os

class UserDatabase:
    def __init__(self, db_path="users.db"):
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Membuat tabel users jika belum ada"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT,
            role TEXT DEFAULT 'user'
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_user(self, username, password, name="", role="user"):
        """Menambahkan pengguna baru ke database"""
        # Hash password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO users (username, password_hash, name, role) VALUES (?, ?, ?, ?)",
                (username, password_hash, name, role)
            )
            conn.commit()
            success = True
        except sqlite3.IntegrityError:
            # Username sudah ada
            success = False
        finally:
            conn.close()
            
        return success
    
    def verify_user(self, username, password):
        """Memverifikasi kredensial pengguna"""
        # Hash password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, username, name, role FROM users WHERE username = ? AND password_hash = ?",
            (username, password_hash)
        )
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                "id": user[0],
                "username": user[1],
                "name": user[2],
                "role": user[3]
            }
        return None
    
    def get_user(self, username):
        """Mendapatkan informasi pengguna berdasarkan username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, username, name, role FROM users WHERE username = ?",
            (username,)
        )
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                "id": user[0],
                "username": user[1],
                "name": user[2],
                "role": user[3]
            }
        return None
    
    def create_admin_if_not_exists(self):
        """Membuat akun admin default jika belum ada"""
        if not self.get_user("admin"):
            self.add_user("admin", "admin123", "Administrator", "admin")