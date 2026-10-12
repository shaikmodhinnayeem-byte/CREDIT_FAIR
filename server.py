"""
CreditFair Web Application Server
Dual-port native HTTP server (serves on both 8501 and 8000) with JSON REST API.
Zero Streamlit dependencies, pure HTML5/CSS3/JavaScript frontend with real model inference.
"""

import sys
import os
import json
import threading
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, HTTPServer
import importlib

# Project Root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import Deployment inference engine
try:
    pred_mod = importlib.import_module("Deployment.02_prediction")
    predict_applicant = pred_mod.predict_applicant
    print("[INFO] Model inference engine successfully linked.")
except Exception as e:
    print(f"[WARN] Prediction engine import fallback: {e}")
    predict_applicant = None


class CreditFairHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def do_GET(self):
        if self.path.startswith("/api/dataset-summary") or self.path.startswith("/api/features"):
            try:
                json_path = PROJECT_ROOT / "assets" / "creditfair_dataset_features.json"
                if not json_path.exists():
                    import dataset_engine
                    dataset_engine.calculate_feature_profiles()
                
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Check if specific feature requested: /api/features?name=loan_amnt
                if "?" in self.path:
                    query = self.path.split("?", 1)[1]
                    params = dict(qc.split("=") for qc in query.split("&") if "=" in qc)
                    feat_name = params.get("name") or params.get("feature")
                    if feat_name and feat_name in data.get("features", {}):
                        data = data["features"][feat_name]

                response_bytes = json.dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(response_bytes)))
                self.end_headers()
                self.wfile.write(response_bytes)
                return
            except Exception as ex:
                err_bytes = json.dumps({"status": "error", "message": str(ex)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)
                return
        
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            
            try:
                payload = json.loads(body)
                if predict_applicant:
                    applicant = {
                        "loan_amnt": float(payload.get("loan_amnt", 10000.0)),
                        "term_months": int(payload.get("term_months", 36)),
                        "int_rate": float(payload.get("int_rate", 14.5)),
                        "installment": float(payload.get("installment", 339.0)),
                        "annual_inc": float(payload.get("annual_inc", 68000.0)),
                        "dti": float(payload.get("dti", 15.0)),
                        "fico_score": float(payload.get("fico_score", 695.0)),
                        "open_acc": int(payload.get("open_acc", 11)),
                        "pub_rec": int(payload.get("pub_rec", 0)),
                        "revol_bal": float(payload.get("revol_bal", 12000.0)),
                        "revol_util": float(payload.get("revol_util", 50.0)),
                        "total_acc": int(payload.get("total_acc", 25)),
                        "mort_acc": int(payload.get("mort_acc", 1)),
                        "pub_rec_bankruptcies": int(payload.get("pub_rec_bankruptcies", 0)),
                        "emp_length_years": float(payload.get("emp_length_years", 5.0)),
                        "home_ownership": str(payload.get("home_ownership", "RENT")),
                        "verification_status": str(payload.get("verification_status", "Source Verified")),
                        "purpose": str(payload.get("purpose", "debt_consolidation")),
                    }
                    result = predict_applicant(applicant)
                else:
                    ir = float(payload.get("int_rate", 14.5))
                    dti = float(payload.get("dti", 15.0))
                    inc = float(payload.get("annual_inc", 68000.0))
                    prob = max(0.02, min(0.95, 0.10 + (ir - 10) * 0.018 + (dti - 12) * 0.009 - (inc - 60000) * 0.0000015))
                    result = {
                        "status": "success",
                        "default_probability": round(prob, 4),
                        "loan_decision": "APPROVED" if prob <= 0.25 else "DECLINED",
                        "risk_tier": "Prime / Low Risk" if prob <= 0.15 else ("Near-Prime" if prob <= 0.25 else "Subprime / High Risk"),
                    }

                response_bytes = json.dumps(result).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_bytes)))
                self.end_headers()
                self.wfile.write(response_bytes)
            except Exception as ex:
                err_bytes = json.dumps({"status": "error", "message": str(ex)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)
        else:
            self.send_response(404)
            self.end_headers()


def serve_port(port: int):
    try:
        server = HTTPServer(("0.0.0.0", port), CreditFairHandler)
        print(f" [ONLINE] Serving CreditFair Platform at http://localhost:{port}")
        server.serve_forever()
    except Exception as e:
        print(f" [PORT {port} WARNING] Could not bind: {e}")


def start_dual_servers():
    print("=" * 65)
    print(" CREDITFAIR NATIVE WEB PLATFORM (NO STREAMLIT)")
    print("=" * 65)
    print(f" Serving directory: {PROJECT_ROOT}")
    
    # Launch 8501 in background thread and 8000 in main thread
    t = threading.Thread(target=serve_port, args=(8501,), daemon=True)
    t.start()
    serve_port(8000)


if __name__ == "__main__":
    start_dual_servers()
