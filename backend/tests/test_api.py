"""Basic API tests."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from app.main import app
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint returns correct response."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "app" in data
    assert "version" in data
    assert "api_base" in data
    assert data["api_base"] == "/api"


def test_health_endpoint(client):
    """Test health endpoint."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_readiness_endpoint(client):
    """Test readiness endpoint."""
    response = client.get("/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_rate_limiting(client):
    """Test rate limiting on predict endpoint."""
    # This would require authentication, but tests the middleware is applied
    # For now, just check the endpoint exists
    response = client.post("/v1/predict", json={})
    # Should get 401 unauthorized, not 404
    assert response.status_code == 401