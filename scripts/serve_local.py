"""Local GPU OpenAI-compatible + interactive chat server for the fine-tuned Sanad QLoRA model."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "qwen2.5-7b-sanad-qlora" / "final_adapter"
DEFAULT_HF_ADAPTER = "3isa2002/qwen2.5-7b-sanad-arabic-qlora"
DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
DEFAULT_MODEL_ID = "sanad-qwen25-ft"


def configure_hf_cache() -> None:
    """Keep Hugging Face weights on A:\\AI-Models\\huggingface and avoid Xet stalls on Windows."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if os.environ.get("HF_HOME"):
        return
    preferred = Path(r"A:\AI-Models\huggingface")
    if preferred.exists():
        os.environ["HF_HOME"] = str(preferred)


class ChatEngine(Protocol):
    model_id: str
    base_model: str
    adapter_source: str
    device_label: str

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> tuple[str, int, int]: ...


@dataclass
class QloraInferenceEngine:
    """Loads 4-bit Qwen2.5-7B + Sanad PEFT LoRA adapter for local RTX GPU inference."""

    model: Any
    tokenizer: Any
    model_id: str
    base_model: str
    adapter_source: str
    device_label: str
    lock: threading.Lock

    @classmethod
    def from_pretrained(
        cls,
        *,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_source: str | None = None,
        model_id: str = DEFAULT_MODEL_ID,
    ) -> "QloraInferenceEngine":
        configure_hf_cache()
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        resolved_adapter = adapter_source
        if not resolved_adapter:
            resolved_adapter = (
                str(DEFAULT_ADAPTER_DIR) if DEFAULT_ADAPTER_DIR.is_dir() else DEFAULT_HF_ADAPTER
            )

        has_cuda = torch.cuda.is_available()
        compute_dtype = torch.float16 if has_cuda else torch.float32
        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            if has_cuda
            else None
        )

        tokenizer = AutoTokenizer.from_pretrained(resolved_adapter)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        load_kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "dtype": compute_dtype,
        }
        if has_cuda:
            load_kwargs["device_map"] = {"": 0}
            if bnb_config is not None and "bnb-4bit" not in base_model.lower():
                load_kwargs["quantization_config"] = bnb_config

        base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        peft_model = PeftModel.from_pretrained(base, resolved_adapter)
        peft_model.eval()

        device_label = torch.cuda.get_device_name(0) if has_cuda else "cpu"
        return cls(
            model=peft_model,
            tokenizer=tokenizer,
            model_id=model_id,
            base_model=base_model,
            adapter_source=resolved_adapter,
            device_label=device_label,
            lock=threading.Lock(),
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> tuple[str, int, int]:
        import torch

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        with self.lock:
            inputs = self.tokenizer(prompt_text, return_tensors="pt")
            target_device = next(self.model.parameters()).device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            prompt_tokens = int(inputs["input_ids"].shape[1])

            do_sample = temperature > 0.05
            gen_kwargs: dict[str, Any] = {
                **inputs,
                "max_new_tokens": max(1, min(max_tokens, 1024)),
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9

            with torch.inference_mode():
                output_ids = self.model.generate(**gen_kwargs)

            generated_ids = output_ids[0][prompt_tokens:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            return text, prompt_tokens, int(generated_ids.shape[0])


class ChatMessage(BaseModel):
    role: str = Field(..., min_length=1)
    content: str = Field(...)


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL_ID)
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=2048)


STANDALONE_HTML = """<!doctype html>
<html lang="ar">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sanad QLoRA — Local Fine-Tuned Chat</title>
  <style>
    :root { color-scheme: dark; --bg:#07070b; --card:#0e0e15; --line:rgba(255,255,255,0.1); --text:#ededf2; --dim:#8a8a9a; --cyan:#22d3ee; --violet:#8b5cf6; --ok:#34d399; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background:var(--bg); color:var(--text); min-height:100vh; display:flex; flex-direction:column; }
    header { padding:16px 24px; border-bottom:1px solid var(--line); display:flex; flex-wrap:wrap; gap:12px; align-items:center; justify-content:space-between; background:var(--card); }
    .badge { font-family:monospace; font-size:12px; padding:4px 10px; border-radius:999px; border:1px solid var(--line); color:var(--cyan); }
    main { max-width:860px; width:100%; margin:0 auto; padding:20px 16px; flex:1; display:flex; flex-direction:column; gap:12px; }
    .presets { display:flex; flex-wrap:wrap; gap:8px; }
    .presets button { cursor:pointer; background:var(--card); border:1px solid var(--line); color:var(--dim); border-radius:999px; padding:6px 12px; font-size:12px; }
    .presets button:hover { color:var(--text); border-color:var(--violet); }
    #log { flex:1; min-height:360px; max-height:60vh; overflow-y:auto; border:1px solid var(--line); border-radius:12px; padding:16px; background:var(--card); display:flex; flex-direction:column; gap:12px; }
    .msg { max-width:85%; padding:10px 14px; border-radius:12px; line-height:1.55; white-space:pre-wrap; font-size:14px; }
    .user { align-self:flex-end; background:rgba(139,92,246,0.18); border:1px solid rgba(139,92,246,0.45); }
    .bot { align-self:flex-start; background:#151520; border:1px solid var(--line); }
    .meta { font-family:monospace; font-size:11px; color:var(--ok); margin-top:4px; }
    form { display:flex; gap:8px; }
    input { flex:1; padding:12px 14px; border-radius:10px; border:1px solid var(--line); background:var(--card); color:var(--text); font-size:14px; }
    button[type=submit] { cursor:pointer; padding:12px 20px; border-radius:10px; border:none; background:linear-gradient(135deg, var(--cyan), var(--violet)); color:#07070b; font-weight:700; }
    a { color:var(--cyan); text-decoration:none; font-size:13px; }
  </style>
</head>
<body>
  <header>
    <div>
      <strong>Sanad QLoRA (Fine-Tuned Qwen2.5-7B · Local GPU)</strong>
      <div style="font-size:12px;color:var(--dim);margin-top:2px;">Direct adapter inference (:8002/v1) · <a href="http://127.0.0.1:8000">Open Full Sanad v3 RAG UI (:8000)</a></div>
    </div>
    <span id="status" class="badge">checking GPU...</span>
  </header>
  <main>
    <div class="presets">
      <button type="button" onclick="usePreset('ar_rag')">RAG مصري مع [1]</button>
      <button type="button" onclick="usePreset('handoff')">فخ عدم التأليف HANDOFF</button>
      <button type="button" onclick="usePreset('general')">سؤال عام حر (بدون RAG)</button>
    </div>
    <div id="log"></div>
    <form id="form">
      <input id="input" dir="auto" placeholder="اكتب رسالتك بالعربي أو الإنجليزي لتجربة الموديل المتدرّب مباشرة..." required />
      <button type="submit">إرسال</button>
    </form>
  </main>
  <script>
    const log = document.getElementById("log");
    const input = document.getElementById("input");
    const statusEl = document.getElementById("status");
    const history = [];
    const PRESETS = {
      ar_rag: "Context:\\n[1] الشحن مجاني للطلبات فوق 500 جنيه من منتجات شحن سوقنا. غير كده المصاريف 35 جنيه داخل القاهرة والجيزة و55 جنيه لباقي المحافظات.\\n\\nQuestion: مصاريف الشحن كام للإسكندرية؟",
      handoff: "Context:\\n[1] الدفع متاح كاش عند الاستلام أو ببطاقة بنكية أونلاين.\\n\\nQuestion: ينفع أقسّط الطلب على 12 شهر بفودافون كاش؟",
      general: "اتكلم معايا بالمصري واشرحلي في 3 نقط الفرق بين QLoRA وRAG في أنظمة دعم العملاء."
    };
    function usePreset(k) { input.value = PRESETS[k]; input.focus(); }
    function addMsg(cls, text, meta) {
      const wrap = document.createElement("div");
      wrap.className = "msg " + cls;
      wrap.setAttribute("dir", "auto");
      wrap.textContent = text;
      if (meta) { const m = document.createElement("div"); m.className = "meta"; m.textContent = meta; wrap.appendChild(m); }
      log.appendChild(wrap);
      log.scrollTop = log.scrollHeight;
    }
    fetch("/health").then(r => r.json()).then(d => { statusEl.textContent = d.model + " · " + d.device; }).catch(() => { statusEl.textContent = "offline"; });
    document.getElementById("form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const txt = input.value.trim();
      if (!txt) return;
      input.value = "";
      addMsg("user", txt);
      history.push({ role: "user", content: txt });
      const t0 = performance.now();
      try {
        const res = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: "sanad-qwen25-ft", messages: history.slice(-8), temperature: 0.1, max_tokens: 384 })
        });
        const data = await res.json();
        const reply = data.choices[0].message.content;
        history.push({ role: "assistant", content: reply });
        const ms = Math.round(performance.now() - t0);
        addMsg("bot", reply, `${data.usage?.completion_tokens ?? 0} tok · ${ms} ms`);
      } catch (err) {
        addMsg("bot", "Error calling local server: " + err);
      }
    });
  </script>
</body>
</html>
"""


def create_local_app(engine: ChatEngine) -> FastAPI:
    """Create the OpenAI-compatible + standalone chat FastAPI server around `engine`."""
    app = FastAPI(title="Sanad QLoRA Local Inference Server", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_private_network=True,
    )
    app.state.engine = engine

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def chat_page() -> str:
        return STANDALONE_HTML

    @app.get("/health")
    def health() -> dict[str, Any]:
        eng: ChatEngine = app.state.engine
        return {
            "status": "ok",
            "model": eng.model_id,
            "base_model": eng.base_model,
            "adapter": eng.adapter_source,
            "device": eng.device_label,
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        eng: ChatEngine = app.state.engine
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": eng.model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": "sanad-qlora",
                    "root": eng.adapter_source,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        eng: ChatEngine = app.state.engine
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        try:
            content, prompt_tok, comp_tok = eng.generate(
                msgs,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": eng.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tok,
                "completion_tokens": comp_tok,
                "total_tokens": prompt_tok + comp_tok,
            },
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Sanad QLoRA locally (OpenAI-compatible + Chat UI).")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=None, help="Local adapter dir or HF repo ID")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    print(f"Loading base model '{args.base_model}' + Sanad QLoRA adapter...")
    engine = QloraInferenceEngine.from_pretrained(
        base_model=args.base_model,
        adapter_source=args.adapter,
        model_id=args.model_id,
    )
    print(f"Loaded '{engine.model_id}' on {engine.device_label} (adapter: {engine.adapter_source})")
    app = create_local_app(engine)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
