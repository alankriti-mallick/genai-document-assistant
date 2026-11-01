
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# Page configuration
st.set_page_config(
    page_title="GenAI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 GenAI Document Assistant")
st.write("Upload a PDF, extract text, create embeddings and store them.")

# Load embedding model only once
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    # Step 1: Extract PDF text
    pdf_reader = PdfReader(uploaded_file)

    extracted_text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            extracted_text += page_text + "\n"

    if extracted_text.strip():

        # Step 2: Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = text_splitter.split_text(extracted_text)

        st.subheader("Document chunks")
        st.write(f"Total chunks: {len(chunks)}")

        with st.expander("View extracted chunks"):
            for index, chunk in enumerate(chunks):
                st.write(f"Chunk {index + 1}")
                st.write(chunk)
                st.divider()

        # Step 3: Create embeddings and store in ChromaDB
        if st.button("Create embeddings and store"):

            with st.spinner("Creating embeddings..."):

                # Load embedding model
                model = load_embedding_model()

                # Convert chunks into numerical vectors
                embeddings = model.encode(
                    chunks,
                    show_progress_bar=False
                )

                # Create persistent ChromaDB database
                client = chromadb.PersistentClient(
                    path="./chroma_db"
                )

                collection = client.get_or_create_collection(
                    name="document_chunks"
                )

                # Clear previous records for this demo
                existing_ids = collection.get()["ids"]

                if existing_ids:
                    collection.delete(ids=existing_ids)

                # Store chunks and embeddings
                collection.add(
                    ids=[
                        f"chunk_{index}"
                        for index in range(len(chunks))
                    ],
                    documents=chunks,
                    embeddings=embeddings.tolist(),
                    metadatas=[
                        {
                            "document_name": uploaded_file.name,
                            "chunk_number": index + 1
                        }
                        for index in range(len(chunks))
                    ]
                )

            st.success("Embeddings created and stored successfully!")

            st.write(
                "Total vectors stored:",
                collection.count()
            )

    else:
        st.warning(
            "No selectable text found in this PDF. "
            "Scanned PDFs may require OCR."
        )
