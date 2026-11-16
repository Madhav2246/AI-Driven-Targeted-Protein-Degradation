#!/usr/bin/env python3
"""
LIGHTWEIGHT LOCAL HTTP SERVER FOR PROTAC PIPELINE WEB DEMO
Serves the interactive research dashboard, datasets, API endpoints, and live simulations.
Usage:
    python web_app/server.py [port]
Defaults to port 8000 (http://localhost:8000).
"""

import os
import sys
import json
import csv
import http.server
import socketserver
import urllib.parse
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.scoring_engine import ProtacPropertyCalculator
from src.docking.ternary_complex import TernaryComplexModeler
from src.docking.ternary_kinetics import TernaryKineticsEngine

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


class ProtacWebHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler with REST API endpoints for live PROTAC exploration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # API Endpoints
        if path == "/api/leads":
            self.send_json_response(self.get_leads())
        elif path == "/api/chembl_data":
            self.send_json_response(self.get_chembl_data())
        elif path == "/api/protac_benchmark":
            self.send_json_response(self.get_protac_benchmarks())
        elif path == "/api/training_history":
            self.send_json_response(self.get_training_history())
        elif path == "/api/ablation_results":
            self.send_json_response(self.get_ablation_results())
        elif path == "/api/length_scan":
            self.send_json_response(self.get_length_scan())
        elif path == "/api/simulate_hook":
            alpha = float(query.get("alpha", [3.65])[0])
            kd_t = float(query.get("kd_t", [50.0])[0])
            kd_e = float(query.get("kd_e", [150.0])[0])
            engine = TernaryKineticsEngine(kd_poi_nm=kd_t, kd_e3_nm=kd_e)
            sim = engine.simulate_dose_response(alpha=alpha)
            # Serialize for JSON
            cleaned_sim = {
                "alpha": sim["alpha"],
                "dc50_nm": sim["dc50_nm"],
                "optimal_conc_nm": sim["optimal_conc_nm"],
                "hook_threshold_nm": sim["hook_threshold_nm"],
                "therapeutic_window_fold": sim["therapeutic_window_fold"],
                "d_max_percent": sim["d_max_percent"],
                "concentrations": [p["p_total_nm"] for p in sim["curve_points"]],
                "ternary_percentages": [round(p["percent_ternary"], 2) for p in sim["curve_points"]],
            }
            self.send_json_response(cleaned_sim)
        else:
            # Fallback to static file server
            super().do_GET()

    def send_json_response(self, data: Any):
        content = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def get_leads(self) -> List[Dict[str, Any]]:
        csv_file = os.path.join(PROJECT_ROOT, "lead_protac_candidates.csv")
        return self._read_csv_as_dict(csv_file)

    def get_chembl_data(self) -> List[Dict[str, Any]]:
        csv_file = os.path.join(PROJECT_ROOT, "data", "chembl_brd4_bioactivities.csv")
        return self._read_csv_as_dict(csv_file)

    def get_protac_benchmarks(self) -> List[Dict[str, Any]]:
        csv_file = os.path.join(PROJECT_ROOT, "data", "protac_db_benchmark.csv")
        return self._read_csv_as_dict(csv_file)

    def get_training_history(self) -> List[Dict[str, Any]]:
        csv_file = os.path.join(PROJECT_ROOT, "logs", "training_history.csv")
        return self._read_csv_as_dict(csv_file)

    def get_ablation_results(self) -> List[Dict[str, Any]]:
        csv_file = os.path.join(PROJECT_ROOT, "experiments", "ablation_study_results.csv")
        return self._read_csv_as_dict(csv_file)

    def get_length_scan(self) -> List[Dict[str, Any]]:
        csv_file = os.path.join(PROJECT_ROOT, "experiments", "linker_length_scan_results.csv")
        return self._read_csv_as_dict(csv_file)

    def _read_csv_as_dict(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)


def run_server(port: int = PORT):
    os.chdir(WEB_DIR)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), ProtacWebHandler) as httpd:
        print("\n" + "=" * 80)
        print("          AI-DRIVEN TPD PROTAC PIPELINE -- WEB INTERFACE DEMO SERVER")
        print("=" * 80)
        print(f"  Authors: Madhav Yalamarthi & Hareesh")
        print(f"  Server URL: http://localhost:{port}")
        print(f"  Interactive Dashboard: http://localhost:{port}/index.html")
        print("  Press Ctrl+C to terminate the local server.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down demo server.")


if __name__ == "__main__":
    run_server()

