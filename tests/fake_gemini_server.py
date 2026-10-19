"""A local stand-in for the Gemini REST API.

It lets the end-to-end tests drive the real SDK over HTTP (request building,
retries, response parsing) without a key or network access. Embeddings are
hashed bags of words; answers quote the first excerpt of the prompt.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from knowledge_assistant.embeddings import LocalHashEmbedder


def _text_of(content: dict) -> str:
    return " ".join(part.get("text", "") for part in content.get("parts", []))


class FakeGemini:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []
        self.fail_next: list[int] = []  # HTTP status codes to return before succeeding
        self.block_next = False  # answer the next generation with a safety block
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> FakeGemini:
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    @staticmethod
    def answer(prompt: str) -> dict:
        excerpt = re.search(r'<excerpt id="(\d+)"[^>]*>\n(.*?)\n</excerpt>', prompt, re.S)
        if "poem" in prompt or not excerpt:
            return {"status": "insufficient_evidence", "statements": [], "note": "Not covered."}
        first_line = excerpt.group(2).split("\n")[-1]
        return {
            "status": "answered",
            "statements": [
                {
                    "text": first_line[:200],
                    "support": "explicit",
                    "sources": [int(excerpt.group(1))],
                }
            ],
            "note": "",
        }

    def _handler(self):
        fake = self
        embedder = LocalHashEmbedder(768)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:  # keep the test output clean
                pass

            def _send(self, status: int, body: dict) -> None:
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _vector(self, content: dict, dimensions: int) -> list[float]:
                vector = embedder.embed_query(_text_of(content)).tolist()
                return (vector + [0.0] * dimensions)[:dimensions]

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.requests.append((self.path, body))
                if fake.fail_next:
                    status = fake.fail_next.pop(0)
                    message = "API key not valid." if status == 400 else "failure"
                    self._send(status, {"error": {"code": status, "message": message}})
                    return
                if self.path.endswith(":generateContent") and fake.block_next:
                    fake.block_next = False
                    self._send(200, {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}})
                elif self.path.endswith(":generateContent"):
                    prompt = _text_of(body["contents"][0])
                    text = json.dumps(fake.answer(prompt))
                    self._send(
                        200,
                        {
                            "candidates": [
                                {
                                    "content": {"role": "model", "parts": [{"text": text}]},
                                    "finishReason": "STOP",
                                }
                            ],
                            "usageMetadata": {
                                "promptTokenCount": len(prompt) // 4,
                                "candidatesTokenCount": len(text) // 4,
                                "totalTokenCount": (len(prompt) + len(text)) // 4,
                            },
                            "modelVersion": "gemini-3.5-flash-lite",
                        },
                    )
                elif self.path.endswith(":batchEmbedContents"):
                    vectors = [
                        {"values": self._vector(r["content"], r.get("outputDimensionality", 768))}
                        for r in body["requests"]
                    ]
                    self._send(200, {"embeddings": vectors})
                elif self.path.endswith(":embedContent"):
                    dimensions = body.get("outputDimensionality", 768)
                    content = body.get("content") or {}
                    self._send(200, {"embedding": {"values": self._vector(content, dimensions)}})
                else:
                    self._send(
                        404, {"error": {"code": 404, "message": f"unknown path {self.path}"}}
                    )

        return Handler
