import streamlit as st
import pymupdf
import faiss
import pickle
import numpy as np
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# STREAMLIT CONFIGURATION

st.set_page_config(
    page_title="Insurance Claim Submission",
    page_icon="📄",
    layout="centered"
)

# BACKGROUND / CSS

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

    input {
        background-color: white !important;
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



# TITLE


st.markdown(
    '<h1 style="color:brown;">Insurance Claim Submission</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p style="color:#0f172a;">'
    'Submit your identification proof and claim bill.'
    '</p>',
    unsafe_allow_html=True
)



# FILE LOCATIONS


CLAIMS_FAISS_FILE = "claims.faiss"

CLAIMS_METADATA_FILE = "claim_metadata.pkl"

COUNTER_FILE = "claim_counter.pkl"

# NEW FILE
CLAIM_STATUS_FILE = "claim_status.pkl"



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


# CLAIM NUMBER GENERATOR


def generate_claim_number():

    """
    Generates sequential claim numbers.

    First claim = 1000
    Next claim = 1001
    Next claim = 1002
    etc.

    No UUID is used.
    """

    if os.path.exists(COUNTER_FILE):

        try:

            with open(
                COUNTER_FILE,
                "rb"
            ) as f:

                last_claim_number = pickle.load(f)

        except Exception:

            last_claim_number = 999

    else:

        last_claim_number = 999


    # Generate next claim number

    claim_number = (
        int(last_claim_number) + 1
    )


    # Save latest claim number

    with open(
        COUNTER_FILE,
        "wb"
    ) as f:

        pickle.dump(
            claim_number,
            f
        )


    return claim_number



# DOCUMENT PROCESSING FUNCTION


def process_document(
    uploaded_file
):

    """
    Reads a PDF, extracts text and creates chunks.
    """

    try:
        # READ PDF
        
        pdf_bytes = uploaded_file.read()

        # OPEN PDF
    
        pdf = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        text = ""

        # EXTRACT TEXT

        for page in pdf:

            text += page.get_text()

        pdf.close()

        # CHECK TEXT
        
        if not text.strip():

            return None

        # CHUNK TEXT
    
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )


        chunks = splitter.split_text(
            text
        )


        if not chunks:

            return None


        return chunks


    except Exception as e:

        st.error(
            f"Error processing document: {e}"
        )

        return None


# SAVE CLAIM STATUS

def save_claim_status(
    claim_number
):

    """
    Saves one claim-level status record.

    Example:

    {
        "claim_number": 1000,
        "claim_status": "Claim Submitted"
    }
    """


    # LOAD EXISTING CLAIM STATUS


    if os.path.exists(
        CLAIM_STATUS_FILE
    ):

        try:

            with open(
                CLAIM_STATUS_FILE,
                "rb"
            ) as f:

                claim_status_data = pickle.load(
                    f
                )

        except Exception:

            claim_status_data = []

    else:

        claim_status_data = []


    
    # CHECK WHETHER CLAIM ALREADY EXISTS
    

    existing_claim = any(
        int(item.get("claim_number", -1))
        == int(claim_number)
        for item in claim_status_data
    )


    
    # ADD NEW CLAIM
    

    if not existing_claim:

        claim_status_data.append(
            {
                "claim_number":
                    int(claim_number),

                "claim_status":
                    "Claim Submitted"
            }
        )


    
    # SAVE CLAIM STATUS FILE
    

    with open(
        CLAIM_STATUS_FILE,
        "wb"
    ) as f:

        pickle.dump(
            claim_status_data,
            f
        )



# CLAIM DETAILS


st.markdown(
    '<h3 style="color:brown;">Claim Details</h3>',
    unsafe_allow_html=True
)


claimant_name = st.text_input(
    "Claimant Name"
)


claim_amount = st.number_input(
    "Claim Amount (Rs.)",
    min_value=0.0,
    step=100.0,
    format="%.2f"
)



# DOCUMENT UPLOAD


st.markdown(
    '<h3 style="color:brown;">Required Documents</h3>',
    unsafe_allow_html=True
)


identification_document = st.file_uploader(
    "Upload Identification Proof",
    type=["pdf"],
    key="identification"
)


bill_document = st.file_uploader(
    "Upload Bill / Claim Supporting Document",
    type=["pdf"],
    key="bill"
)



# SUBMIT CLAIM


if st.button(
    "Submit Claim"
):

    
    # VALIDATION
    

    if not claimant_name.strip():

        st.error(
            "Please enter claimant name."
        )

        st.stop()


    if claim_amount <= 0:

        st.error(
            "Please enter a valid claim amount."
        )

        st.stop()


    if identification_document is None:

        st.error(
            "Please upload identification proof."
        )

        st.stop()


    if bill_document is None:

        st.error(
            "Please upload the claim bill."
        )

        st.stop()


    
    # GENERATE CLAIM NUMBER
    

    claim_number = generate_claim_number()


    
    # PROCESS IDENTIFICATION DOCUMENT
    

    identification_chunks = process_document(
        identification_document
    )


    if identification_chunks is None:

        st.error(
            "Could not extract text from "
            "identification proof."
        )

        st.stop()


    
    # PROCESS BILL DOCUMENT
    

    bill_chunks = process_document(
        bill_document
    )


    if bill_chunks is None:

        st.error(
            "Could not extract text from "
            "the bill document."
        )

        st.stop()


    
    # COMBINE CHUNKS
    

    all_chunks = (
        identification_chunks
        +
        bill_chunks
    )


    
    # CREATE EMBEDDINGS
    

    try:

        embeddings = model.embed_documents(
            all_chunks
        )


        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

    except Exception as e:

        st.error(
            f"Could not create embeddings: {e}"
        )

        st.stop()


    
    # NORMALIZE EMBEDDINGS
    

    faiss.normalize_L2(
        embeddings
    )


    
    # LOAD / CREATE FAISS INDEX
    

    try:

        if os.path.exists(
            CLAIMS_FAISS_FILE
        ):

            index = faiss.read_index(
                CLAIMS_FAISS_FILE
            )


            # Check dimensions

            if index.d != embeddings.shape[1]:

                st.error(
                    "Embedding dimension does not match "
                    "the existing FAISS index."
                )

                st.stop()

        else:

            index = faiss.IndexFlatIP(
                embeddings.shape[1]
            )

    except Exception as e:

        st.error(
            f"Could not load FAISS index: {e}"
        )

        st.stop()


    
    # GET STARTING FAISS INDEX
    

    start_index = index.ntotal


    
    # ADD EMBEDDINGS TO FAISS
    

    index.add(
        embeddings
    )


    
    # SAVE FAISS INDEX
    

    try:

        faiss.write_index(
            index,
            CLAIMS_FAISS_FILE
        )

    except Exception as e:

        st.error(
            f"Could not save FAISS index: {e}"
        )

        st.stop()


    
    # LOAD EXISTING CLAIM METADATA
    

    if os.path.exists(
        CLAIMS_METADATA_FILE
    ):

        try:

            with open(
                CLAIMS_METADATA_FILE,
                "rb"
            ) as f:

                metadata = pickle.load(
                    f
                )

        except Exception as e:

            st.error(
                f"Could not load claim metadata: {e}"
            )

            st.stop()

    else:

        metadata = []


    
    # SAVE IDENTIFICATION METADATA
    

    for i, chunk in enumerate(
        identification_chunks
    ):

        global_index = (
            start_index + i
        )


        metadata.append(
            {
                "claim_number":
                    claim_number,

                "claimant_name":
                    claimant_name,

                "claim_amount":
                    claim_amount,

                "document_type":
                    "identification",

                "document_file":
                    identification_document.name,

                "chunk_id":
                    i,

                "faiss_index":
                    global_index,

                "text":
                    chunk,

                "embedding":
                    embeddings[i].tolist()
            }
        )


    
    # SAVE BILL METADATA
    

    bill_start_index = (
        start_index
        +
        len(identification_chunks)
    )


    for i, chunk in enumerate(
        bill_chunks
    ):

        embedding_position = (
            len(identification_chunks)
            +
            i
        )


        global_index = (
            bill_start_index + i
        )


        metadata.append(
            {
                "claim_number":
                    claim_number,

                "claimant_name":
                    claimant_name,

                "claim_amount":
                    claim_amount,

                "document_type":
                    "bill",

                "document_file":
                    bill_document.name,

                "chunk_id":
                    i,

                "faiss_index":
                    global_index,

                "text":
                    chunk,

                "embedding":
                    embeddings[
                        embedding_position
                    ].tolist()
            }
        )


    
    # SAVE CLAIM METADATA
    

    try:

        with open(
            CLAIMS_METADATA_FILE,
            "wb"
        ) as f:

            pickle.dump(
                metadata,
                f
            )

    except Exception as e:

        st.error(
            f"Could not save claim metadata: {e}"
        )

        st.stop()


    
    # SAVE CLAIM STATUS
    

    try:

        save_claim_status(
            claim_number
        )

    except Exception as e:

        st.error(
            f"Claim documents were saved, but "
            f"claim status could not be saved: {e}"
        )

        st.stop()


    
    # SUCCESS MESSAGE
    

    st.markdown(
        """
        <p style="
            color:green;
            font-size:22px;
            font-weight:bold;
        ">
            Claim Submitted Successfully
        </p>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <p style="color:brown;">
            <b>Claim Number:</b> {claim_number}
        </p>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <p style="color:brown;">
            <b>Claimant Name:</b> {claimant_name}
        </p>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <p style="color:brown;">
            <b>Claim Amount:</b>
            Rs. {claim_amount:,.2f}
        </p>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <p style="color:brown;">
            <b>Claim Status:</b>
            Claim Submitted
        </p>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <p style="
            color:green;
            font-weight:bold;
        ">
            Claim status has been saved successfully.
        </p>
        """,
        unsafe_allow_html=True
    )
