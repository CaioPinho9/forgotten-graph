from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from db import init_db, load_node_adjacency_payload


class GraphRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/adjacency":
            return super().do_GET()

        query = parse_qs(parsed.query)
        node_id = (query.get("node_id") or [""])[0].strip()
        if not node_id:
            self.send_error(HTTPStatus.BAD_REQUEST, "missing query param: node_id")
            return

        payload = load_node_adjacency_payload(node_id)
        if payload is None:
            self.send_error(HTTPStatus.NOT_FOUND, "node not found")
            return

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=300")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Forgotten Graph viewer + API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    init_db()
    httpd = ThreadingHTTPServer((args.host, args.port), GraphRequestHandler)
    print(f"Serving on http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
