import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import webbrowser
import threading
import time

PORT = 8050

def main():
    print("==================================================")
    print(" 🔥 Madhya Pradesh Forest Fire Prediction System ")
    print("==================================================")
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    sys.path.insert(0, project_dir)
    
    # Check for required model files
    required_artifacts = [
        "model.pkl", "rf_model.pkl", "scaler.pkl", 
        "features.pkl", "thresholds.pkl", "ensemble_weight.pkl"
    ]
    missing = [f for f in required_artifacts if not os.path.exists(f)]
    if missing:
        print(f"[ERROR] Missing model artifacts: {missing}")
        sys.exit(1)
        
    print("[SUCCESS] Model artifacts verified.")
    print("[INFO] Initializing FastAPI application & dataset...")
    
    import app as forest_app
    
    # Open browser automatically after server starts
    def open_browser():
        time.sleep(2.0)
        url = f"http://127.0.0.1:{PORT}"
        print(f"[INFO] Opening web interface at {url} ...")
        webbrowser.open(url)
        
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        import uvicorn
        print(f"[LAUNCH] Starting Forest Fire Prediction Server on http://127.0.0.1:{PORT} ...")
        uvicorn.run(forest_app.app, host="127.0.0.1", port=PORT, log_level="info")
    except ImportError:
        print("[ERROR] uvicorn is not installed. Run 'pip install -r requirements.txt'")
    except Exception as e:
        print(f"[ERROR] Server error: {e}")

if __name__ == "__main__":
    main()
