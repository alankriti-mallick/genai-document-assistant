import streamlit as st

# Page configuration
st.set_page_config(
    page_title="GenAI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# Application title
st.title("🤖 GenAI Document Assistant")

st.write(
    "Upload your documents and ask questions "
    "to understand their content."
)

# Document upload
st.subheader("Upload your document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

# Display uploaded file information
if uploaded_file is not None:

    st.success("Document uploaded successfully!")

    st.write("**File name:**", uploaded_file.name)

    st.write(
        "**File size:**",
        round(uploaded_file.size / 1024, 2),
        "KB"
    )

    st.write("We will process this document in the next step.")

else:
    st.info("Please upload a PDF document to get started.")