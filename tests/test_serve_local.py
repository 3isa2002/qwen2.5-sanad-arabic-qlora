from fastapi.testclient import TestClient

from scripts.serve_local import create_local_app


class FakeEngine:
    model_id = "sanad-qwen25-ft"
    base_model = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    adapter_source = "outputs/qwen2.5-7b-sanad-qlora/final_adapter"
    device_label = "RTX 2070 Test"

    def __init__(self):
        self.calls = []

    def generate(self, messages, *, temperature=0.1, max_tokens=512):
        self.calls.append((messages, temperature, max_tokens))
        return "الشحن بـ 35 جنيه جوه القاهرة [1].", 18, 9


def test_serve_local_openai_endpoints_and_ui():
    engine = FakeEngine()
    client = TestClient(create_local_app(engine))

    assert client.get("/").status_code == 200
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model"] == "sanad-qwen25-ft"

    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "sanad-qwen25-ft"

    chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "sanad-qwen25-ft",
            "messages": [{"role": "user", "content": "مصاريف الشحن كام؟"}],
            "temperature": 0.1,
            "max_tokens": 256,
        },
    )
    assert chat.status_code == 200
    body = chat.json()
    assert body["choices"][0]["message"]["content"] == "الشحن بـ 35 جنيه جوه القاهرة [1]."
    assert body["usage"]["total_tokens"] == 27
    assert len(engine.calls) == 1
