"""Standalone script to launch the Gradio Chat UI for arXiv Paper Curator."""

import logging
from src.config import settings
from src.gradio_app import build_demo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Launching arXiv Paper Curator Gradio Chat UI on port %d...", settings.gradio_port)
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=settings.gradio_port,
        share=False,
    )
