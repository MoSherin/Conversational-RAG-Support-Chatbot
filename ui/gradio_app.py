"""Standalone Gradio chat interface for the conversational RAG API."""

from __future__ import annotations

import logging
import uuid
import httpx

import gradio as gr


logger = logging.getLogger(__name__)

CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
#hero { padding: 1.6rem 1.8rem; border-radius: 20px; margin: 1rem 0 1.2rem; background: linear-gradient(125deg, #111827 0%, #263b68 58%, #4969aa 100%); color: white; }
#hero h1 { margin: 0 0 .4rem; color: white; font-size: clamp(1.65rem, 4vw, 2.25rem); }
#hero p { margin: 0; color: #dbe6ff; font-size: 1rem; }
#chat-panel { border: 1px solid var(--border-color-primary); border-radius: 18px; padding: .75rem; }
#send-button { min-width: 105px; align-self: stretch; }
#status { min-height: 1.5rem; }
footer { display: none !important; }
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size="lg",
)


def build_demo(api_base_url: str) -> gr.Blocks:
    """Build the existing UI; chat and reset calls go to the backend API."""
    api_base_url = api_base_url.rstrip("/")

    def send_message(message: str, history: list[dict] | None, conversation_id: str):
        question = (message or "").strip()
        if not question:
            return history or [], "", "Type a question to get started."

        conversation_id = conversation_id or uuid.uuid4().hex
        try:
            response = httpx.post(
                f"{api_base_url}/chat",
                json={"question": question, "conversation_id": conversation_id},
                timeout=300.0,
            )
            response.raise_for_status()
            answer = response.json()["answer"]
            updated_history = list(history or [])
            updated_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            return updated_history, "", "Answer ready."
        except Exception:
            logger.exception("Gradio chat request failed")
            return history or [], question, "I couldn't get a response. Check the backend and model service, then try again."

    def start_new_chat():
        new_id = uuid.uuid4().hex
        return [], "", new_id, new_id, "New conversation started."

    def clear_current_chat(conversation_id: str):
        conversation_id = conversation_id or uuid.uuid4().hex
        try:
            response = httpx.post(
                f"{api_base_url}/reset",
                json={"conversation_id": conversation_id},
                timeout=30.0,
            )
            response.raise_for_status()
            return [], "", conversation_id, conversation_id, "Conversation cleared."
        except Exception:
            logger.exception("Could not reset the Gradio conversation")
            return None, "", conversation_id, conversation_id, "Couldn't clear this conversation. Check the backend."

    with gr.Blocks(title="Conversational RAG") as demo:
        conversation_id = gr.State(value=lambda: uuid.uuid4().hex)

        gr.HTML(
            """
            <div id="hero">
              <h1>Conversational RAG</h1>
              <p>Ask a question, explore the knowledge base, and follow up naturally.</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=480, elem_id="chat-panel"):
                chatbot = gr.Chatbot(
                    height=510,
                    show_label=False,
                    buttons=["copy", "copy_all"],
                    placeholder="<strong>Start a conversation</strong><br>Ask something about your knowledge base.",
                )
                with gr.Row(equal_height=True):
                    message = gr.Textbox(
                        placeholder="Ask a question… (Enter to send, Shift+Enter for a new line)",
                        show_label=False,
                        lines=2,
                        max_lines=6,
                        scale=8,
                        autofocus=True,
                    )
                    send = gr.Button("Send", variant="primary", elem_id="send-button", scale=1)

                with gr.Row():
                    new_chat = gr.Button("＋ New chat", variant="secondary")
                    reset_chat = gr.Button("Clear this chat", variant="stop")
                status = gr.Markdown("Ready when you are.", elem_id="status")

            with gr.Column(scale=1, min_width=245):
                gr.Markdown("### Try asking")
                gr.Markdown("Choose a starter, or write your own question.")
                gr.Examples(
                    examples=[
                        "Summarize the main points in the knowledge base.",
                        "What does the document say about pricing?",
                        "Explain one of the key topics in more detail.",
                        "Can you clarify that?",
                    ],
                    inputs=message,
                    examples_per_page=4,
                    label="Suggested questions",
                )
                gr.Markdown(
                    "---\n"
                    "**Your conversation**\n\n"
                    "Messages in this chat share one conversation ID. Use **New chat** to start a separate thread, or **Clear this chat** to erase its saved memory."
                )
                conversation_id_view = gr.Textbox(
                    label="Conversation ID",
                    interactive=False,
                    buttons=["copy"],
                )

        outputs = [chatbot, message, status]
        inputs = [message, chatbot, conversation_id]
        send.click(send_message, inputs=inputs, outputs=outputs)
        message.submit(send_message, inputs=inputs, outputs=outputs)
        new_chat.click(
            start_new_chat,
            inputs=None,
            outputs=[chatbot, message, conversation_id, conversation_id_view, status],
        )
        reset_chat.click(
            clear_current_chat,
            inputs=conversation_id,
            outputs=[chatbot, message, conversation_id, conversation_id_view, status],
        )
        demo.load(lambda cid: cid, inputs=conversation_id, outputs=conversation_id_view)

    return demo
