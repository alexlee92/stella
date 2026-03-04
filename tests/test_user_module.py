import unittest
from user_module import UserManager, User

class TestUserManager(unittest.TestCase):
    def setUp(self):
        self.user_manager = UserManager()

    def test_add_user(self):
        self.assertTrue(self.user_manager.add_user("testuser", "test@example.com", "password123"))
        with self.assertRaises(ValueError):
            self.user_manager.add_user("testuser", "invalid-email", "password123")
        with self.assertRaises(ValueError):
            self.user_manager.add_user("testuser", "test@example.com", "password123")

    def test_get_user_by_username(self):
        self.user_manager.add_user("testuser", "test@example.com", "password123")
        user = self.user_manager.get_user_by_username("testuser")
        self.assertEqual(user.username, "testuser")
        with self.assertRaises(ValueError):
            self.user_manager.get_user_by_username("nonexistent")

    def test_update_user(self):
        self.user_manager.add_user("testuser", "test@example.com", "password123")
        self.assertTrue(self.user_manager.update_user("testuser", email="newemail@example.com"))
        self.assertTrue(self.user_manager.update_user("testuser", password="newpassword123"))
        with self.assertRaises(ValueError):
            self.user_manager.update_user("nonexistent")

    def test_delete_user(self):
        self.user_manager.add_user("testuser", "test@example.com", "password123")
        self.assertTrue(self.user_manager.delete_user("testuser"))
        self.assertFalse(self.user_manager.delete_user("nonexistent"))

    def test_list_users(self):
        self.user_manager.add_user("testuser1", "test1@example.com", "password123")
        self.user_manager.add_user("testuser2", "test2@example.com", "password123")
        users = self.user_manager.list_users()
        self.assertEqual(len(users), 2)
        self.assertIn({"username": "testuser1", "email": "test1@example.com"}, users)
        self.assertIn({"username": "testuser2", "email": "test2@example.com"}, users)

if __name__ == '__main__':
    unittest.main()
