#Ncert Bot

RAG (Retrieval-Augmented Generation) Chatbot is a conversational AI application designed to answer questions based on text extracted from PDF files. Leveraging LangChain, Google Generative AI, and FAISS vector stores, this chatbot processes document-based queries in a highly accurate and responsive manner.

**Live Demo**: [RAG Chatbot on Render](https://rag-chatbot-ebp7.onrender.com)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

### PDF Text Extraction with OCR Support
The chatbot extracts text from uploaded PDFs, with an OCR fallback to handle scanned or image-based PDFs. It uses `PyPDF2` for initial text extraction and `pytesseract` for OCR processing on non-text pages.

### Text Chunking for Optimal Processing
To ensure efficient question-answering, the text is chunked into smaller sections. This is done using LangChain’s `RecursiveCharacterTextSplitter`, enabling the chatbot to analyze lengthy documents more effectively.

### Google AI Embeddings and FAISS Vector Store
The chatbot generates document embeddings using Google Generative AI, storing them in a FAISS vector database. This structure allows for fast similarity searches, retrieving the most relevant document chunks based on user questions.

### Conversational AI with LangChain
Using LangChain’s Question Answering Chain, the chatbot responds to user queries by referencing relevant document sections, generating answers within the context of the provided PDFs.

### Image and Link Retrieval (Optional)
For enhanced responses, the chatbot can retrieve relevant images and external links using Google’s Custom Search API.

## Installation

### Prerequisites
- Python 3.8+
- PostgreSQL for user and feedback data storage

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/raushan22882917/rag_chatbo.git
   cd rag_chatbo
