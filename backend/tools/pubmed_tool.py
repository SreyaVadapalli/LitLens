from Bio import Entrez
import json
import os
from dotenv import load_dotenv
load_dotenv()
Entrez.email = os.getenv("NCBI_EMAIL")

Entrez.email = "your@email.com"

def get_pubmed_papers(query: str, max_results: int = 10) -> list:
    try:
        # Search PubMed
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results
        )
        record = Entrez.read(handle)
        handle.close()

        pmids = record["IdList"]
        if not pmids:
            return []

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

        return papers

    except Exception as e:
        print(f"PubMed search error: {e}")
        return []