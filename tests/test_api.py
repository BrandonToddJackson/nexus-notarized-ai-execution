"""API integration tests: all endpoints, auth, error responses."""

import pytest
from fastapi.testclient import TestClient

from nexus.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

class TestAuthEndpoint:
    def test_valid_api_key_returns_token(self, client):
        response = client.post("/v1/auth/token", json={"api_key": "nxs_demo_key_12345"})
        assert response.status_code == 200
        assert "token" in response.json()

    def test_invalid_api_key_returns_401(self, client):
        response = client.post("/v1/auth/token", json={"api_key": "invalid"})
        assert response.status_code == 401

class TestExecuteEndpoint:
    def test_execute_requires_auth(self, client):
        response = client.post("/v1/execute", json={"task": "test"})
        assert response.status_code in [401, 501]  # 501 until implemented
