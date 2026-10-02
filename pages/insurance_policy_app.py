# STREAMLIT FRONT END - INSURANCE POLICY UPLOADER

import streamlit as st
import pickle
import numpy as np
import os
import pymupdf
import faiss

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# STREAMLIT CONFIGURATION

st.set_page_config(
    page_title="Insurance Policy Uploader",
    page_icon="📄",
    layout="centered"
)



# BLUE BACKGROUND


st.markdown(
    """
    <style>

    .stApp {
        background-color: #dbeafe;
    }

    .main {
        background-color: #dbeafe;
    }

    h1 {
        color: #0f172a;
    }

    label {
        color: #0f172a !important;
    }

    [data-testid="stFileUploader"] {
        background-color: #bfdbfe;
        border-radius: 10px;
        padding: 10px;
    }

    .stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# PAGE TITLE


st.markdown(
    '<h1 style="color:brown;">Insurance Policy Uploader</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p style="color:#0f172a;">'
    'Upload the insurance company policy PDF. '
    'Only the latest policy is maintained.'
    '</p>',
    unsafe_allow_html=True
)


# FILE LOCATIONS

FAISS_FILE = "insurance_policy.faiss"
METADATA_FILE = "insurance_policy.pkl"
PDF_FILE = "insurance_policy.pdf"

# EMBEDDING MODEL

#@st.cache_resource
#def load_embedding_model():

#    return HuggingFaceEmbeddings(
#        model_name="BAAI/bge-small-en-v1.5"
#    )

#model = load_embedding_model()

model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# PDF UPLOAD

file = st.file_uploader(
    "Upload Insurance Company Policy",
    type=["pdf"]
)

# PROCESS POLICY

if file:

    if st.button("Update Insurance Policy"):

        # 1. DELETE EXISTING POLICY FILES
        
        try:

            if os.path.exists(FAISS_FILE):
                os.remove(FAISS_FILE)

            if os.path.exists(METADATA_FILE):
                os.remove(METADATA_FILE)

            if os.path.exists(PDF_FILE):
                os.remove(PDF_FILE)

        except Exception as e:

            st.error(
                f"Could not remove existing policy files: {e}"
            )

            st.stop()

        # 2. SAVE NEW POLICY PDF
        
        try:

            with open(PDF_FILE, "wb") as f:

                f.write(
                    file.getbuffer()
                )

        except Exception as e:

            st.error(
                f"Could not save insurance policy PDF: {e}"
            )

            st.stop()

        # 3. READ PDF
        
        try:

            pdf = pymupdf.open(PDF_FILE)

            pages = []

            for page_number, page in enumerate(pdf):

                page_text = page.get_text()

                if page_text.strip():

                    pages.append(
                        page_text
                    )

            pdf.close()

        except Exception as e:

            st.error(
                f"Could not read insurance policy PDF: {e}"
            )

            if os.path.exists(PDF_FILE):
                os.remove(PDF_FILE)

            st.stop()

        # 4. EXTRACT TEXT
        
        text = "\n\n".join(pages)

        if not text.strip():

            st.error(
                "Could not extract text from the insurance policy PDF."
            )

            if os.path.exists(PDF_FILE):
                os.remove(PDF_FILE)

            st.stop()

        # 5. CHUNK POLICY
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_text(text)


        if not chunks:

            st.error(
                "No chunks were created from the insurance policy."
            )

            if os.path.exists(PDF_FILE):
                os.remove(PDF_FILE)

            st.stop()

        # 6. CREATE EMBEDDINGS
        
        try:

            embeddings = model.embed_documents(
                chunks
            )

            embeddings = np.asarray(
                embeddings,
                dtype="float32"
            )

        except Exception as e:

            st.error(
                f"Could not create embeddings: {e}"
            )

            if os.path.exists(PDF_FILE):
                os.remove(PDF_FILE)

            st.stop()

        # 7. NORMALIZE EMBEDDINGS
        
        faiss.normalize_L2(
            embeddings
        )

        # 8. CREATE NEW FAISS INDEX
        
        try:

            dimension = embeddings.shape[1]

            index = faiss.IndexFlatIP(
                dimension
            )

            index.add(
                embeddings
            )

        except Exception as e:

            st.error(
                f"Could not create FAISS index: {e}"
            )

            if os.path.exists(PDF_FILE):
                os.remove(PDF_FILE)

            st.stop()

        # 9. SAVE FAISS INDEX
        
        try:

            faiss.write_index(
                index,
                FAISS_FILE
            )

        except Exception as e:

            st.error(
                f"Could not save FAISS index: {e}"
            )

            if os.path.exists(PDF_FILE):
                os.remove(PDF_FILE)

            st.stop()

        # 10. CREATE POLICY METADATA
        
        policy_metadata = []

        for i, chunk in enumerate(chunks):

            policy_metadata.append(
                {
                    "policy_file": file.name,

                    "chunk_id": i,

                    "faiss_index": i,

                    "text": chunk
                }
            )

        # 11. SAVE METADATA AS PICKLE

        try:

            with open(
                METADATA_FILE,
                "wb"
            ) as f:

                pickle.dump(
                    policy_metadata,
                    f
                )

        except Exception as e:

            st.error(
                f"Could not save policy metadata: {e}"
            )

            # Remove partially created files
            if os.path.exists(FAISS_FILE):
                os.remove(FAISS_FILE)

            if os.path.exists(PDF_FILE):
                os.remove(PDF_FILE)

            st.stop()

        # 12. SUCCESS MESSAGE
        
        st.markdown(
            """
            <p style="
                color:green;
                font-size:22px;
                font-weight:bold;
            ">
                Insurance Policy Updated Successfully
            </p>
            """,
            unsafe_allow_html=True
        )

        # 13. DISPLAY INFORMATION - Should not be displayed
        
        #st.markdown(
        ##    f"""
        #    <p style="color:brown;">
        #        <b>Policy File:</b> {file.name}
        #    </p>
        #    """,
        #    unsafe_allow_html=True
        #)

        #st.markdown(
        #    f"""
        #    <p style="color:brown;">
        #        <b>Total Chunks:</b> {len(chunks)}
        #    </p>
        #    """,
        #    unsafe_allow_html=True
        #)

        #st.markdown(
        #    f"""
        #    <p style="color:brown;">
        #        <b>Embedding Dimension:</b> {embeddings.shape[1]}
        #    </p>
        #    """,
        #    unsafe_allow_html=True
        #)

        #st.markdown(
        #    f"""
        #    <p style="color:brown;">
        #        <b>FAISS Vectors:</b> {index.ntotal}
        #    </p>
        #    """,
        #    unsafe_allow_html=True
        #)
