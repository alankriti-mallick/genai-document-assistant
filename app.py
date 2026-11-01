import streamlit as st
import ollama

from src.pdf_processor import process_pdf
from src.embeddings import get_embedding_model, generate_embeddings
from src.vector_store import get_collection, store_chunks, search_chunks


st.set_page_config(page_title="GenAI Document Assistant")

st.title("GenAI Document Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

model = get_embedding_model()
collection = get_collection()

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if st.button("Process PDF"):

        chunks, metadatas = process_pdf(uploaded_file)

        embeddings = generate_embeddings(model, chunks)

        # Remove previous chunks from this document
        existing = collection.get(
            where={"document_name": uploaded_file.name}
        )

        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        store_chunks(collection, chunks, embeddings, metadatas)

        st.success("PDF processed successfully!")


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


question = st.chat_input("Ask something about your document")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    if collection.count() == 0:
        st.warning("Please upload and process a PDF first.")

    else:
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                question_embedding = generate_embeddings(
                    model, [question]
                )[0]

                results = search_chunks(
                    collection, question_embedding
                )

                retrieved_chunks = results["documents"][0]
                retrieved_metadata = results["metadatas"][0]

                context = "\n\n".join(retrieved_chunks)

                system_message = {
                    "role": "system",
                    "content": f"""
                    You are a helpful document assistant.

                    Answer only using the provided document context.
                    If the answer is not available, say you cannot find it
                    in the document.

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

                st.write(answer)

                # Display sources
                with st.expander("View sources"):
                    for metadata in retrieved_metadata:
                        st.write(
                            f"{metadata.get('document_name', 'Unknown document')} "
                            f"- Page {metadata.get('page_number', 'Unknown')}"
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })


if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()