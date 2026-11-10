from __future__ import annotations

import os

from ui.gradio_app import build_demo

demo = build_demo(os.getenv("BACKEND_URL", "http://backend:8000"))

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=False,
    )
