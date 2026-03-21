import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "uptime_seconds" in data


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "schema_validity_rate_pct" in data
    assert "avg_masking_latency_ms" in data


def test_chat_completions_passthrough():
    payload = {
        "model": "qwen-2.5-7b-instruct",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) > 0
    assert "content" in data["choices"][0]["message"]


def test_chat_completions_structured_json():
    payload = {
        "model": "qwen-2.5-7b-instruct",
        "messages": [{"role": "user", "content": "Extract customer record for John Doe"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "CustomerRecord",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "integer"},
                        "tier": {"type": "string", "enum": ["SILVER", "GOLD", "PLATINUM"]},
                    },
                    "required": ["customer_id", "tier"],
                },
            },
        },
        "temperature": 0.5,
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    parsed = json.loads(content)
    assert "customer_id" in parsed
    assert isinstance(parsed["customer_id"], int)
    assert parsed["tier"] in ["SILVER", "GOLD", "PLATINUM"]


def test_chat_completions_streaming():
    payload = {
        "model": "qwen-2.5-7b-instruct",
        "messages": [{"role": "user", "content": "Stream response"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "Ping",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["OK", "ERROR"]},
                    },
                    "required": ["status"],
                },
            },
        },
        "stream": True,
    }

    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        chunks = []
        for line in response.iter_lines():
            if line.startswith("data: ") and not line.endswith("[DONE]"):
                chunk_data = json.loads(line[6:])
                chunks.append(chunk_data)

        assert len(chunks) > 0
