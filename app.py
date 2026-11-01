
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# --------------------------------------------------
# 1. Streamlit Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="GenAI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 GenAI Document Assistant")
st.write("Upload a PDF and ask questions about its content.")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# 2. Load Embedding Model
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 3. Initialize ChromaDB
# --------------------------------------------------

@st.cache_resource
def get_chroma_collection():

    client = chromadb.PersistentClient(path="./chroma_db")

    return client.get_or_create_collection(
        name="document_chunks"
    )


model = load_embedding_model()
collection = get_chroma_collection()


# --------------------------------------------------
# 4. PDF Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    # Extract text page by page
    pdf_reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):

        page_text = page.extract_text()

        if page_text and page_text.strip():

            pages.append({
                "page_number": page_number,
                "text": page_text
            })

    if pages:

        # --------------------------------------------------
        # 5. Text Chunking with Page Metadata
        # --------------------------------------------------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = []
        metadatas = []

        for page in pages:

            page_chunks = text_splitter.split_text(page["text"])

            for chunk in page_chunks:

                chunks.append(chunk)

                metadatas.append({
                    "document_name": uploaded_file.name,
                    "page_number": page["page_number"]
                })

        st.write(f"Total pages: {len(pages)}")
        st.write(f"Total chunks: {len(chunks)}")

        # --------------------------------------------------
        # 6. Generate Embeddings and Store in ChromaDB
        # --------------------------------------------------

        if st.button("Create embeddings and store"):

            with st.spinner("Processing document..."):

                embeddings = model.encode(chunks)

                document_id = uploaded_file.name.replace(" ", "_")

                ids = [
                    f"{document_id}_{index}"
                    for index in range(len(chunks))
                ]

                # Remove older chunks for the same document
                existing = collection.get(
                    where={"document_name": uploaded_file.name}
                )

                if existing["ids"]:
                    collection.delete(ids=existing["ids"])

                collection.add(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas
                )

            st.success("Document processed successfully!")

    else:

        st.warning("No readable text found in this PDF.")


# --------------------------------------------------
# 7. Chat Interface
# --------------------------------------------------

st.subheader("💬 Chat with your document")

# Display previous conversation
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


question = st.chat_input("Ask a question about your document...")


if question:

    # Save and display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    if collection.count() == 0:

        st.warning("Please upload and process a document first.")

    else:

        with st.chat_message("assistant"):

            with st.spinner("Searching document and generating answer..."):

                # --------------------------------------------------
                # 8. Retrieve Relevant Chunks
                # --------------------------------------------------

                question_embedding = model.encode(question)

                results = collection.query(
                    query_embeddings=[
                        question_embedding.tolist()
                    ],
                    n_results=min(3, collection.count())
                )

                retrieved_chunks = results["documents"][0]
                retrieved_metadata = results["metadatas"][0]
                retrieved_distances = results["distances"][0]

                # Experimental relevance threshold
                DISTANCE_THRESHOLD = 1.25

                # Keep only relevant chunks
                filtered_results = [
                    (chunk, metadata, distance)
                    for chunk, metadata, distance in zip(
                        retrieved_chunks,
                        retrieved_metadata,
                        retrieved_distances
                    )
                    if distance <= DISTANCE_THRESHOLD
                ]

                retrieved_chunks = [item[0] for item in filtered_results]
                retrieved_metadata = [item[1] for item in filtered_results]
                retrieved_distances = [item[2] for item in filtered_results]

                if not retrieved_chunks:

                    answer = (
                        "I couldn't find sufficiently relevant information "
                        "in the document to answer your question."
                    )

                else:
                    # Keep your existing context creation and Ollama generation here

                    # Combine retrieved chunks
                    context = "\n\n".join(retrieved_chunks)

                    # --------------------------------------------------
                    # 9. Generate Answer Using Ollama
                    # --------------------------------------------------

                    system_message = {
                        "role": "system",
                        "content": f"""
                        You are a helpful document assistant.

                        Answer questions using the provided document context
                        and previous conversation.

                        Use conversation history to understand follow-up questions.

                        Answer using only information supported by the document.

                        If the answer is unavailable in the context,
                        clearly say that the document does not contain
                        the required information.

                        Do not invent facts.

                        Document context:
                        {context}
                        """
                    }

                    chat_messages = [
                        system_message
                    ] + st.session_state.messages

                    response = ollama.chat(
                        model="llama3.2:3b",
                        messages=chat_messages
                    )

                    answer = response["message"]["content"]

                    # Display answer
                    st.write(answer)

                    # --------------------------------------------------
                    # 10. Display Source References
                    # --------------------------------------------------

                    st.markdown("**📚 Sources**")

                    # Avoid repeating the same page reference
                    displayed_sources = set()

                    for metadata in retrieved_metadata:

                        source = (
                            metadata.get("document_name", "Unknown document"),
                            metadata.get("page_number", "Page information unavailable")
                        )

                        if source not in displayed_sources:

                            displayed_sources.add(source)

                            st.caption(
                                f"📄 {source[0]} — Page {source[1]}"
                            )

                    with st.expander("View retrieved document sections"):

                        for index, chunk in enumerate(retrieved_chunks):

                            st.markdown(f"**Source {index + 1}**")
                            st.write(chunk)

                            st.divider()

                    # Save AI response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })


# --------------------------------------------------
# 11. Clear Conversation
# --------------------------------------------------

if st.button("Clear conversation"):

    st.session_state.messages = []

    st.rerun()
