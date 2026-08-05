from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_metadata_endpoint():
    response = client.get("/api/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "types" in data
    assert "races" in data
    assert "attributes" in data
    assert "monster_types" in data