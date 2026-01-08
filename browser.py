#!/usr/bin/env python3
"""
HTTP server for the Extremal Cograph Viewer.
Serves per-parameter cache files from browser_d3_cache/ for on-the-fly loading.
"""

import http.server
import json
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import socketserver

SCRIPT_DIR = Path(__file__).parent
PORT = 8765
CACHE_DIR = SCRIPT_DIR / "browser_d3_cache"


def ensure_cache_exists():
    """Check if cache exists, rebuild if not."""
    index_file = CACHE_DIR / "index.json"

    if not CACHE_DIR.exists() or not index_file.exists():
        print("Cache not found, rebuilding...")
        generate_script = SCRIPT_DIR / "generate_browser_cache.py"
        try:
            result = subprocess.run(
                [sys.executable, str(generate_script)],
                cwd=str(SCRIPT_DIR),
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode != 0:
                print(f"Warning: Cache generation failed: {result.stderr}")
                return False
            print("Cache rebuilt successfully")
        except Exception as e:
            print(f"Warning: Error rebuilding cache: {e}")
            return False

    return True


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler serving per-parameter graph data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SCRIPT_DIR), **kwargs)

    def handle(self):
        """Handle requests with BrokenPipe error suppression."""
        try:
            super().handle()
        except BrokenPipeError:
            pass  # Client disconnected, ignore
        except ConnectionResetError:
            pass  # Client reset connection, ignore

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/index":
            self._handle_index()
        elif parsed.path == "/api/params":
            self._handle_params()
        elif parsed.path == "/":
            self.send_response(302)
            self.send_header("Location", f"/extremal_viewer_d3.html?v={int(time.time())}")
            self.end_headers()
        else:
            super().do_GET()

    def _handle_index(self):
        """Return the index of available parameters."""
        try:
            index_file = CACHE_DIR / "index.json"
            if index_file.exists():
                with open(index_file, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "Index not found"}')
        except BrokenPipeError:
            pass

    def _handle_params(self):
        """Return data for specific s,t parameters."""
        try:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            s = query.get('s', [None])[0]
            t = query.get('t', [None])[0]

            if s is None or t is None:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "Missing s or t parameter"}')
                return

            cache_file = CACHE_DIR / f"K{s}_{t}.json"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "Parameters not found"}')
        except BrokenPipeError:
            pass

    def log_message(self, format, *args):
        # Suppress routine logging
        if args and isinstance(args[0], str):
            if "/api/" in args[0] or args[0].startswith("GET /extremal"):
                return
        super().log_message(format, *args)


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main():
    # Ensure cache exists
    print("Checking cache...")
    if not ensure_cache_exists():
        print("Warning: Cache may be incomplete")
    print()

    with ReusableTCPServer(("", PORT), ViewerHandler) as httpd:
        url = f"http://localhost:{PORT}/"
        print(f"Starting Extremal Cograph Viewer at {url}")
        print("Press Ctrl+C to stop the server")

        # Open browser after a short delay
        def open_browser():
            webbrowser.open(url)

        threading.Timer(0.5, open_browser).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    main()
