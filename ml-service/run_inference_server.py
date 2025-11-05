"""
Run the vessel arrival prediction inference server

Usage:
    python run_inference_server.py
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Get configuration from environment
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    reload = os.getenv('API_RELOAD', 'false').lower() == 'true'

    print("=" * 70)
    print("TYTOALBA VESSEL ARRIVAL PREDICTION SERVICE")
    print("=" * 70)
    print(f"Starting server on http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"Alternative Docs: http://{host}:{port}/redoc")
    print(f"Auto-reload: {reload}")
    print("=" * 70)

    # Run server
    uvicorn.run(
        "api_inference:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
