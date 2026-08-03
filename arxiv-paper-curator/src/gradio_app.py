"""Gradio Chat Interface for arXiv Paper Curator RAG Pipeline.

Provides interactive question-answering UI with real-time streaming,
filter controls, and expandable source paper citations.
"""

import logging
from typing import AsyncGenerator

import gradio as gr

from .config import settings
from .schemas.rag import AskRequest
from .services.rag import RAGPipeline

logger = logging.getLogger(__name__)


def create_rag_pipeline() -> RAGPipeline:
    """Initialize RAGPipeline instance for Gradio interface."""
    return RAGPipeline()


pipeline = create_rag_pipeline()


async def rag_chat_stream(
    question: str,
    mode: str,
    category: str,
    top_k: int,
) -> AsyncGenerator[tuple[str, str], None]:
    """Process user question through RAG pipeline and yield streaming answer + sources HTML."""
    if not question.strip():
        yield "Please enter a valid research question.", ""
        return

    req = AskRequest(
        query=question,
        mode=mode,
        category=category.strip() if category and category.strip() else None,
        top_k=top_k,
    )

    accumulated_text = ""
    sources_html = ""

    async for item in pipeline.answer_stream(req):
        event_type = item.get("event")
        data = item.get("data")

        if event_type == "token":
            accumulated_text += str(data)
            yield accumulated_text, sources_html
        elif event_type == "metadata":
            if isinstance(data, dict):
                raw_sources = data.get("sources", [])
                took_ms = data.get("took_ms", 0)
                used_count = data.get("used_chunk_count", 0)

                if not raw_sources:
                    sources_html = (
                        "<div style='margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-left: 4px solid #ffc107; color: #333;'>"
                        "<strong>No relevant context chunks found.</strong>"
                        "</div>"
                    )
                else:
                    cards = []
                    for idx, src in enumerate(raw_sources, start=1):
                        paper_id = src.get("paper_id", "N/A")
                        title = src.get("title", "Untitled")
                        section = src.get("section_name", "N/A")
                        score = src.get("relevance_score", 0.0)
                        pdf_url = src.get("pdf_url", "")
                        snippet = src.get("snippet", "")

                        pdf_link = (
                            f"<a href='{pdf_url}' target='_blank' style='color: #0066cc; text-decoration: underline;'>[PDF]</a>"
                            if pdf_url
                            else ""
                        )

                        cards.append(
                            f"<details style='margin-bottom: 8px; border: 1px solid #e0e0e0; border-radius: 4px; padding: 8px; background-color: #ffffff; color: #222222;'>"
                            f"<summary style='font-weight: bold; cursor: pointer; font-size: 14px; color: #1a0dab;'>"
                            f"Source {idx}: [{paper_id}] {title} — § {section} (Score: {score:.4f})"
                            f"</summary>"
                            f"<div style='margin-top: 8px; font-size: 13px; color: #333333;'>"
                            f"<p><strong>Section:</strong> {section} &nbsp;&nbsp; {pdf_link}</p>"
                            f"<blockquote style='margin: 4px 0; padding-left: 10px; border-left: 3px solid #0066cc; font-style: italic; background-color: #f9f9f9; color: #333333;'>"
                            f"{snippet}"
                            f"</blockquote>"
                            f"</div>"
                            f"</details>"
                        )

                    header_meta = f"<p style='font-size: 12px; color: #666;'>Retrieved {len(raw_sources)} source chunks ({used_count} used in context, took {took_ms:.0f}ms)</p>"
                    sources_html = (
                        f"<div style='margin-top: 15px; font-family: sans-serif;'>"
                        f"<h3>📚 Source Attributions</h3>"
                        f"{header_meta}"
                        f"{''.join(cards)}"
                        f"</div>"
                    )

                yield accumulated_text, sources_html


def build_demo() -> gr.Blocks:
    """Construct Gradio UI Blocks layout."""
    with gr.Blocks(title="arXiv Paper Curator — RAG Assistant") as demo:
        gr.Markdown(
            "# 🔬 arXiv Paper Curator — Grounded RAG Assistant\n"
            "Ask questions about computer science and AI research papers. "
            "Answers are strictly grounded in indexed arXiv paper chunks."
        )

        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Research Question",
                    placeholder="e.g. How does attention mechanism work in Transformers?",
                    lines=3,
                )
                submit_btn = gr.Button("Ask Question", variant="primary")

            with gr.Column(scale=1):
                mode_dropdown = gr.Dropdown(
                    choices=["hybrid", "semantic", "keyword"],
                    value="hybrid",
                    label="Retrieval Mode",
                    info="Hybrid combines BM25 keyword + kNN vector search via RRF fusion",
                )
                category_input = gr.Textbox(
                    label="Filter Category (Optional)",
                    placeholder="e.g. cs.AI or cs.CL",
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=8,
                    step=1,
                    label="Context Chunks (Top-K)",
                )

        answer_output = gr.Markdown(label="Generated Answer")
        sources_output = gr.HTML(label="Source Citations")

        submit_btn.click(
            fn=rag_chat_stream,
            inputs=[question_input, mode_dropdown, category_input, top_k_slider],
            outputs=[answer_output, sources_output],
        )

    return demo
