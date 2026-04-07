from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, filename: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    return [
        {"text": chunk, "source": filename, "chunk_id": i}
        for i, chunk in enumerate(chunks)
    ]