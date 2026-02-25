import os
import jwt
from datetime import datetime, timedelta, UTC
from typing import Dict, Optional

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "")
if not SECRET_KEY:
    raise ValueError(
        "JWT_SECRET_KEY environment variable is required. "
        "Set it with: export JWT_SECRET_KEY=<your-secret>"
    )

ALGORITHM = "HS256"


def encode_token(
    payload: Dict[str, object], expires_delta: Optional[timedelta] = None
) -> str:
    to_encode = payload.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Dict[str, object]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")


if __name__ == "__main__":
    # Example usage with Flask — development only
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/login", methods=["POST"])
    def login():
        user_data = request.json  # Assume username and password are validated
        access_token_expires = timedelta(minutes=30)
        access_token = encode_token(user_data, expires_delta=access_token_expires)
        return jsonify(access_token=access_token)

    @app.route("/protected", methods=["GET"])
    def protected():
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"message": "Token is missing"}), 401
        try:
            payload = decode_token(token)
        except ValueError as e:
            return jsonify({"message": str(e)}), 403
        return jsonify(payload)

    app.run(debug=True)
