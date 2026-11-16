"""Live dashboard for the trading agent.

    python -m dashboard                      # then open http://127.0.0.1:8050

Serves a single page that polls ``/api/events`` for new lines of the agent's
event log (``data/agent_events.jsonl``, written by ``python -m agents.main``).
Standard library only.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

STATIC = Path(__file__).parent / "static"
MAX_EVENTS_PER_RESPONSE = 5000


def read_events(path: Path, offset: int) -> dict:
    """Events appended since byte ``offset``. ``reset`` is set when a new session restarted the file."""
    if not path.exists():
        return {"offset": 0, "reset": offset > 0, "events": []}
    size = path.stat().st_size
    reset = size < offset
    if reset:
        offset = 0
    with path.open("rb") as f:
        f.seek(offset)
        chunk = f.read()
    complete = chunk[:chunk.rfind(b"\n") + 1]          # ignore a half-written last line
    events = []
    for line in complete.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"offset": offset + len(complete), "reset": reset,
            "events": events[-MAX_EVENTS_PER_RESPONSE:]}


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, events_path: Path, **kwargs):
        self.events_path = events_path
        super().__init__(*args, **kwargs)

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/events":
            try:
                offset = int(parse_qs(url.query).get("offset", ["0"])[0])
            except ValueError:
                offset = 0
            payload = read_events(self.events_path, max(0, offset))
            self._send(json.dumps(payload).encode(), "application/json")
        else:
            self._send(b"not found", "text/plain", 404)

    def log_message(self, format: str, *args) -> None:       # keep the terminal quiet
        pass


def main() -> None:
    p = argparse.ArgumentParser(description="Live dashboard for the trading agent.")
    p.add_argument("--events", default="data/agent_events.jsonl")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8050)
    args = p.parse_args()

    server = ThreadingHTTPServer((args.host, args.port),
                                 partial(Handler, events_path=Path(args.events)))
    print(f"Dashboard on http://{args.host}:{args.port}  (reading {args.events}, Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
