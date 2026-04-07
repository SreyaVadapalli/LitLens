import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from Bio import Entrez
import os
from dotenv import load_dotenv
load_dotenv()
Entrez.email = os.getenv("NCBI_EMAIL")

Entrez.email = "your@email.com"

app = Server("pubmed-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_pubmed",
            description="Search PubMed for relevant biomedical papers",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for PubMed"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_pubmed":
        query = arguments["query"]
        max_results = arguments.get("max_results", 10)

        # Search PubMed
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results
        )
        record = Entrez.read(handle)
        handle.close()

        pmids = record["IdList"]

        # Fetch details
        handle = Entrez.efetch(
            db="pubmed",
            id=pmids,
            rettype="abstract",
            retmode="xml"
        )
        records = Entrez.read(handle)
        handle.close()

        papers = []
        for record in records["PubmedArticle"]:
            try:
                article = record["MedlineCitation"]["Article"]
                title = str(article["ArticleTitle"])
                pmid = str(record["MedlineCitation"]["PMID"])

                authors = []
                if "AuthorList" in article:
                    for author in article["AuthorList"][:3]:
                        if "LastName" in author:
                            authors.append(str(author["LastName"]))

                papers.append({
                    "title": title,
                    "pmid": pmid,
                    "authors": authors,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
            except Exception:
                continue

        return [TextContent(
            type="text",
            text=json.dumps(papers, indent=2)
        )]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())