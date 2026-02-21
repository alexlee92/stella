import pytest
from flask import Flask, json
from users.api import user_bp


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(user_bp)
    with app.test_client() as client:
        yield client


def test_create_user_nominal(client):
    response = client.post(
        "/users",
        data=json.dumps({"email": "test@example.com", "password": "password123"}),
        content_type="application/json",
    )
    assert response.status_code == 201
    user_data = json.loads(response.data)
    assert "id" in user_data
    assert user_data["email"] == "test@example.com"


def test_create_user_edge_case(client):
    response = client.post(
        "/users",
        data=json.dumps({"email": "test@example.com"}),
        content_type="application/json",
    )
    assert response.status_code == 400
    error_data = json.loads(response.data)
    assert error_data["error"] == "Missing required fields"
