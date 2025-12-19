#!/usr/bin/env python3
"""
HTTP server for the Extremal Cograph Viewer.
Provides folder selection API and serves the visualization.
"""

import http.server
import json
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import socketserver

SCRIPT_DIR = Path(__file__).parent
PORT = 8765


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with folder selection API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SCRIPT_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/folders":
            self._handle_list_folders()
        elif parsed.path == "/api/current":
            self._handle_get_current()
        elif parsed.path == "/":
            # Redirect to the viewer
            self.send_response(302)
            self.send_header("Location", "/extremal_viewer_d3.html")
            self.end_headers()
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/select-folder":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}
            self._handle_select_folder(data.get("folder"), force=data.get("force", False))
        else:
            self.send_error(404)

    def _handle_list_folders(self):
        """List available export folders."""
        folders = []
        for path in SCRIPT_DIR.iterdir():
            if path.is_dir() and path.name.startswith("exports"):
                json_files = list(path.glob("extremal_K*.json"))
                if json_files:
                    cache_file = path / "visualization_cache.json"
                    folders.append({
                        "name": path.name,
                        "path": str(path.relative_to(SCRIPT_DIR)),
                        "file_count": len(json_files),
                        "has_cache": cache_file.exists()
                    })

        folders.sort(key=lambda x: x["name"])

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(folders).encode())

    def _handle_get_current(self):
        """Get current data status."""
        data_file = SCRIPT_DIR / "extremal_graphs.json"
        has_data = data_file.exists() and data_file.stat().st_size > 10

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"has_data": has_data}).encode())

    def _handle_select_folder(self, folder, force=False):
        """Select folder and load/regenerate data."""
        if not folder:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "No folder specified"}).encode())
            return

        folder_path = SCRIPT_DIR / folder
        if not folder_path.exists():
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Folder not found"}).encode())
            return

        cache_file = folder_path / "visualization_cache.json"
        used_cache = False

        # Check if cache exists and we're not forcing regeneration
        if cache_file.exists() and not force:
            used_cache = True
            print(f"Using cached data from: {cache_file}")
        else:
            print(f"Generating new data for: {folder_path} (force={force}, cache_exists={cache_file.exists()})")
            # Run generate_d3_data.py
            generate_script = SCRIPT_DIR / "generate_d3_data.py"
            try:
                result = subprocess.run(
                    [sys.executable, str(generate_script), str(folder_path)],
                    cwd=str(SCRIPT_DIR),
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode != 0:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": "Generation failed",
                        "details": result.stderr
                    }).encode())
                    return

            except subprocess.TimeoutExpired:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Generation timed out"}).encode())
                return

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
                return

        # Read and return the cache file
        try:
            with open(cache_file) as f:
                data = json.load(f)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": True,
                "folder": folder,
                "used_cache": used_cache,
                "graph_count": len(data),
                "data": data
            }).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Failed to read cache: {e}"}).encode())

    def log_message(self, format, *args):
        # Suppress routine logging
        if "/api/" in args[0] or args[0].startswith("GET /extremal"):
            return
        super().log_message(format, *args)


def main():
    with socketserver.TCPServer(("", PORT), ViewerHandler) as httpd:
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
