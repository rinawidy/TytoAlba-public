"""
Run the ML Service API Server
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))

    print("=" * 50)
    print("Starting TytoAlba ML Service")
    print("=" * 50)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"API Docs: http://localhost:{port}/docs")
    print("=" * 50)

    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=True  # Enable auto-reload for development
    )
