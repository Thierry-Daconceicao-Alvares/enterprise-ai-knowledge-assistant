# Deployment

## Local

See the [installation section](../README.md#8-installation) of the README. The index is stored in `data/index` and the uploaded files in `data/storage`; both are ignored by Git.

## Docker

```bash
docker build -t knowledge-assistant .
docker run -p 8501:8501 \
  -e GEMINI_API_KEY=your-key \
  -v ka-index:/app/data/index -v ka-files:/app/data/storage \
  knowledge-assistant
```

The image is based on `python:3.12-slim`, runs as a non-root user, exposes port 8501 and defines a health check on `/_stcore/health`. The CI workflow builds it and checks that the container starts on every push.

To run the HTTP API instead of the web interface:

```bash
docker run -p 8000:8000 -e GEMINI_API_KEY=your-key knowledge-assistant \
  uvicorn knowledge_assistant.api:app --host 0.0.0.0 --port 8000
```

## Public demo on Streamlit Community Cloud

The public demo runs the same code with `PUBLIC_DEMO=true`:

- the 32 sample documents are indexed once when the application starts, in memory;
- each visitor gets a private copy of that index: files they upload are visible only in their session and disappear when it ends (sessions idle for one hour are deleted);
- each session is limited to 30 questions and 3 uploads (configurable);
- the page warns visitors not to upload confidential documents, because the free tier of the Gemini API may use content to improve Google's products.

Steps:

1. Sign in to [share.streamlit.io](https://share.streamlit.io) with the GitHub account that owns the repository.
2. Create an app from the repository, branch `main`, main file `app/streamlit_app.py`, Python 3.12 (advanced settings).
3. In the app's secrets, add:

   ```toml
   GEMINI_API_KEY = "your-key"
   PUBLIC_DEMO = "true"
   ```

4. Deploy. Streamlit installs `requirements.txt`, which installs the package with its pinned dependencies.

Use a key from a Google Cloud project without billing, so that the demo can only use the free tier and cannot generate costs. The demo embeds the 239 passages of the sample documents when it starts, then sends one embedding and one generation request per question. When the daily free-tier quota is used up, visitors are told when the demo will answer again (the quota resets at midnight Pacific time). Quotas are counted per Google Cloud project: an evaluation run with the same key reduces what is left for visitors that day, while a key from a separate project keeps the demo independent.

## Production rollout plan

| Area | Prototype | Production |
|---|---|---|
| Hosting | Single container | Container platform (Cloud Run, Azure Container Apps, Kubernetes) with at least two replicas |
| Identity | None | SSO through the corporate identity provider; roles from directory groups |
| Permissions | None | Document-level access lists stored as metadata and applied in the retrieval filter |
| Vector store | Embedded Chroma on a local disk | Managed vector database or Chroma server with backups |
| Documents | Uploaded by hand | Connector to the document management system, incremental synchronisation of new and withdrawn versions |
| Model API | Gemini free tier | Paid or enterprise endpoint with a data processing agreement, regional data residency if required |
| Secrets | Environment variables | Secret manager with rotation |
| Monitoring | JSON logs | Dashboards for latency, errors, abstention rate, cost; alerts on error rate and quota |
| Quality | Evaluation run by hand | Evaluation in CI on every change of prompt, model, chunking or corpus, with thresholds that block a release |
| Feedback | None | Thumbs up/down with a reason, reviewed weekly with the document owners |

Rollout steps: pilot with one department (purchasing) and its documents → review of the evaluation and of user feedback with the process owners → permissions and SSO → extension to HR and IT documents → general availability.
