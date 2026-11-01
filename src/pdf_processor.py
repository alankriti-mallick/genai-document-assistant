from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "page_number": page_number,
                "text": text
            })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []
    metadatas = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append(chunk)
            metadatas.append({
                "document_name": uploaded_file.name,
                "page_number": page["page_number"]
            })

    return chunks, metadatas