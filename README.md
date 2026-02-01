---
title: Insurance Brochure RAG
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Insurance Brochure RAG Application

An AI-powered RAG system for analyzing insurance brochures and documents.

## Local Development

1. Install requirements: `pip install -r requirements.txt`
2. Set `.env` variables (see `.env.example`)
3. Run `python app.py`

## Hugging Face Deployment

This app is configured to run on Hugging Face Spaces using the **Docker SDK**.

### Configuration
- **Instance Type**: 16GB RAM (CPU Basic) or higher.
- **Secrets**: Add `GROQ_API_KEY` in the Space settings.
