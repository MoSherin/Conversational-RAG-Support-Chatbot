# Container layout

The Compose stack builds three images:

- `ui`: the existing Gradio interface, calling the API over HTTP.
- `backend`: FastAPI, LangGraph, retrieval, and Chroma orchestration.
- `models`: embedding and reranker inference plus a chat-model gateway. The embedding and reranker weights are baked into this image.

The model service uses Gemini when `RAG_PROVIDER=gemini`. With `RAG_PROVIDER=local`, it calls an Ollama server at `OLLAMA_BASE_URL`; Ollama itself is not bundled into these three images. The current project does not yet implement an Azure-hosted model provider.

## Build and start

1. Create or update `.env` and set `RAG_PROVIDER` to exactly `gemini` or `local`. For Gemini, set `GOOGLE_API_KEY`. The existing `.env` currently has a provider value the application rejects, as shown in the VS Code terminal.
2. Build and start the stack with `docker compose up --build`.
3. Open Gradio at `http://localhost:7860`, the API docs at `http://localhost:8000/docs`, and the health endpoint at `http://localhost:8000/health`.

The existing `chroma_db` folder is bind-mounted so the index is retained. The document folder is mounted read-only. The models image is large because it includes both transformer models. The first image build downloads these weights.

Do not commit `.env`; it may contain provider and LangSmith credentials. For Azure, put secrets in the deployment platform's secret store and replace the local bind mounts with persistent Azure storage before production deployment.
