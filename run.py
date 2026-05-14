"""
Entry point for the MAPE-K Fraud Detection system.
"""
import sys
import os
import socket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PORT = int(os.environ.get("PORT", 8000))

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("0.0.0.0", port)) == 0

if __name__ == "__main__":
    import uvicorn

    if is_port_in_use(PORT):
        print(f"WARNING: Port {PORT} is already in use. Using port {PORT + 1} instead.")
        PORT += 1

    print("=" * 60)
    print("  MAPE-K Agentic Fraud Detection API")
    print(f"  Starting server on http://localhost:{PORT}")
    print("=" * 60)
    print()
    print("  ENDPOINTS:")
    print(f"    POST http://localhost:{PORT}/transaction/process")
    print(f"    POST http://localhost:{PORT}/feedback")
    print(f"    GET  http://localhost:{PORT}/transaction/{{id}}")
    print(f"    GET  http://localhost:{PORT}/patterns")
    print(f"    GET  http://localhost:{PORT}/adversarial/{{id}}")
    print(f"    GET  http://localhost:{PORT}/analytics")
    print()
    uvicorn.run("api.server:app", host="0.0.0.0", port=PORT, reload=True)
