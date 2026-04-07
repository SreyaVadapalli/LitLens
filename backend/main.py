from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import fitz
import os
from rag.ingestion import chunk_text
from rag.retriever import store_chunks
from agent.nodes import summarize_paper, compare_papers, extract_topic_and_search_pubmed
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="LitLens", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "LitLens API is running ✅"}

@app.post("/upload")
async def upload_papers(files: List[UploadFile] = File(...)):
    summaries = []

    for file in files:
        contents = await file.read()

        # Save to temp file
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Parse with PyMuPDF
        doc = fitz.open(temp_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        num_pages = doc.page_count
        doc.close()

        # Clean up temp file
        os.remove(temp_path)

        # Chunk and store
        chunks = chunk_text(full_text, file.filename)
        store_chunks(chunks)

        # Summarize with Claude
        summary = summarize_paper(file.filename)

        summaries.append({
            "filename": file.filename,
            "num_pages": num_pages,
            "num_chunks": len(chunks),
            "summary": summary
        })

    # Compare papers if more than one
    comparison = None
    if len(summaries) > 1:
        comparison = compare_papers(summaries)

    # Search PubMed via MCP
    pubmed_papers = extract_topic_and_search_pubmed(summaries)

    return {
        "papers": summaries,
        "comparison": comparison,
        "pubmed_recommendations": pubmed_papers
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)