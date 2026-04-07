import anthropic
import os
from dotenv import load_dotenv
from rag.retriever import retrieve_chunks

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def summarize_paper(filename: str) -> str:
    chunks = retrieve_chunks(filename, n_results=8)
    context = "\n\n".join(chunks)

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": f"""You are a biomedical research assistant. 
Summarize this research paper clearly and concisely.

Paper content:
{context}

Provide:
1. Study objective
2. Methods used
3. Key findings
4. Limitations
5. Evidence level (RCT, cohort, systematic review, etc.)"""
            }
        ]
    )
    return message.content[0].text


def compare_papers(summaries: list) -> dict:
    summaries_text = "\n\n".join([
        f"Paper {i+1}: {s['filename']}\n{s['summary']}"
        for i, s in enumerate(summaries)
    ])

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": f"""You are a biomedical research assistant.
Compare these research papers and return ONLY a JSON object with no extra text.

Papers:
{summaries_text}

Return this exact JSON structure:
{{
    "similarities": ["similarity 1", "similarity 2"],
    "differences": ["difference 1", "difference 2"],
    "evidence_levels": {{"paper_filename": "evidence level"}},
    "research_gaps": ["gap 1", "gap 2"],
    "consensus": "overall consensus statement"
}}"""
            }
        ]
    )

    import json
    text = message.content[0].text
    clean = text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)

def extract_topic_and_search_pubmed(summaries: list) -> list:
    from tools.pubmed_tool import get_pubmed_papers

    # Use Claude to extract the core topic
    summaries_text = "\n\n".join([
        f"Paper {i+1}: {s['filename']}\n{s['summary'][:300]}"
        for i, s in enumerate(summaries)
    ])

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": f"""Based on these paper summaries, generate a 
concise PubMed search query (max 8 words).
Return ONLY the search query, nothing else.

Summaries:
{summaries_text}"""
            }
        ]
    )

    query = message.content[0].text.strip()
    print(f"PubMed search query: {query}")

    # Search PubMed via MCP
    papers = get_pubmed_papers(query, max_results=10)
    return papers