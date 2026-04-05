# MediBot AI

MediBot AI is a smart medical chatbot web application built to provide users with helpful health-related information through a clean and modern interface. The project is designed to work with an LLM-powered backend so users can ask medical questions and receive AI-generated responses in real time.

## Preview

![MediBot UI](./assets/medibot-ui.png)

> Save your screenshot inside an `assets` folder and name it `medibot-ui.png` so this image shows correctly in GitHub README.

---

## Features

- Beautiful and modern medical chatbot UI
- Clean 3-column layout
- Quick health topic shortcuts
- Session statistics panel
- Specialties section
- Emergency warning section
- Chat input with real-time response area
- Easy integration with Flask / FastAPI backend
- Designed for LLM-powered healthcare assistant projects

---

## Project Objective

The main goal of this project is to build an AI-powered medical assistant that helps users ask health-related questions in natural language and receive useful responses using a Large Language Model.

This project combines:
- **Frontend** for user interaction
- **Backend API** for processing prompts
- **LLM integration** for generating answers
- **Vector database / embeddings** for medical knowledge retrieval

---

## Tech Stack

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Python
- Flask

### AI / LLM Tools
- LangChain
- OpenAI
- HuggingFace Embeddings
- Pinecone Vector Database

---

## Project Structure

```bash
medibot_ai/
│
├── app.py
├── setup.py
├── requirements.txt
├── .env
├── README.md
│
├── data/
│   └── medical_pdf_files.pdf
│
├── research/
│   └── trials.ipynb
│
└── src/
    ├── __init__.py
    ├── helper.py
    ├── prompt.py
