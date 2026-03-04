import re
import hashlib
from typing import List, Dict

class User:
    def __init__(self, username: str, email: str, password: str):
        self.username = username
        self.email = email
        self.password_hash = self._hash_password(password)

    @staticmethod
    def _is_valid_email(email: str) -> bool:
        return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

    @staticmethod
    def _hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

class UserManager:
    def __init__(self):
        self.users = []

    def add_user(self, username: str, email: str, password: str) -> bool:
        if not User._is_valid_email(email):
            raise ValueError("Invalid email address")
        
        for user in self.users:
            if user.username == username or user.email == email:
                return False

        new_user = User(username, email, password)
        self.users.append(new_user)
        return True

    def get_user_by_username(self, username: str) -> User:
        for user in self.users:
            if user.username == username:
                return user
        raise ValueError("User not found")

    def update_user(self, username: str, email: str = None, password: str = None) -> bool:
        user = self.get_user_by_username(username)
        
        if email and User._is_valid_email(email):
            user.email = email
        
        if password:
            user.password_hash = User._hash_password(password)
        
        return True

    def delete_user(self, username: str) -> bool:
        for i, user in enumerate(self.users):
            if user.username == username:
                del self.users[i]
                return True
        return False

    def list_users(self) -> List[Dict]:
        return [{"username": user.username, "email": user.email} for user in self.users]
