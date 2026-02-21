from flask import Blueprint, request, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

from .models import User, Base

engine = create_engine("sqlite:///users.db", connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
user_bp = Blueprint('users', __name__)

@user_bp.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    hashed_password = generate_password_hash(data['password'], method='bcrypt')
    new_user = User(email=data['email'], hashed_password=hashed_password, created_at=datetime.datetime.now(datetime.timezone.utc))
    session = Session()
    try:
        session.add(new_user)
        session.commit()
        return jsonify(new_user.to_dict()), 201
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 400

@user_bp.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    session = Session()
    user = session.query(User).filter_by(id=user_id).first()
    if user:
        return jsonify(user.to_dict()), 200
    else:
        return jsonify({'error': 'User not found'}), 404

@user_bp.route('/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    data = request.get_json()
    session = Session()
    user = session.query(User).filter_by(id=user_id).first()
    if user:
        user.email = data.get('email', user.email)
        if 'password' in data:
            user.hashed_password = generate_password_hash(data['password'], method='bcrypt')
        try:
            session.commit()
            return jsonify(user.to_dict()), 200
        except Exception as e:
            session.rollback()
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'User not found'}), 404

@user_bp.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    session = Session()
    user = session.query(User).filter_by(id=user_id).first()
    if user:
        try:
            session.delete(user)
            session.commit()
            return jsonify({'message': 'User deleted'}), 200
        except Exception as e:
            session.rollback()
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'User not found'}), 404

@user_bp.route('/users/login', methods=['POST'])
def login_user():
    data = request.get_json()
    email = data['email']
    password = data['password']
    session = Session()
    user = session.query(User).filter_by(email=email).first()
    if user and check_password_hash(user.hashed_password, password):
        return jsonify({'message': 'Login successful', 'user': user.to_dict()}), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@user_bp.route('/users/logout', methods=['POST'])
def logout_user():
    # Typically handled by session management in a real application
    return jsonify({'message': 'Logout successful'}), 200