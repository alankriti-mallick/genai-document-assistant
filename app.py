import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="GenAI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 GenAI Document Assistant")
st.write("Upload a PDF and extract its text.")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    # Read the PDF from the uploaded file
    pdf_reader = PdfReader(uploaded_file)

    # Extract text from each page
    extracted_text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            extracted_text += page_text + "\n"

    if extracted_text.strip():
        st.subheader("Extracted text")
        st.text_area(
            "Text from your PDF",
            extracted_text,
            height=300
        )
        
        # Split the extracted text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = text_splitter.split_text(extracted_text)

        st.subheader("Text chunks")
        st.write(f"Created {len(chunks)} chunks")

        for index, chunk in enumerate(chunks):
            with st.expander(f"Chunk {index + 1}"):
                st.write(chunk)
    else:
        st.warning(
            "No selectable text was found. "
            "This may be a scanned PDF that needs OCR."
        )