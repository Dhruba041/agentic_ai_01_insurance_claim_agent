import streamlit as st
import pickle
import os
import pandas as pd

# STREAMLIT CONFIGURATION

st.set_page_config(
    page_title="Insurance Claim Summary",
    page_icon="📋",
    layout="wide"
)


# FILE LOCATIONS

CLAIM_STATUS_FILE = "claim_status.pkl"
CLAIMS_METADATA_FILE = "claim_metadata.pkl"



# CSS

# CSS

st.markdown(
    """
    <style>

    .stApp {
        background-color: #dbeafe;
    }

    .main {
        background-color: #dbeafe;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #7f1d1d !important;
    }

    /* Normal text */
    p, span, div, label {
        color: #7f1d1d;
    }

    /* Streamlit labels */
    label {
        color: #7f1d1d !important;
    }

    /* Selectbox text */
    [data-baseweb="select"] {
        color: #7f1d1d !important;
        background-color: white !important;
    }

    [data-baseweb="select"] * {
        color: #7f1d1d !important;
    }

    /* Selectbox dropdown */
    [role="option"] {
        color: #7f1d1d !important;
        background-color: white !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        background-color: white;
    }

    [data-testid="stDataFrame"] * {
        color: #7f1d1d !important;
    }

    /* Table text */
    [data-testid="stDataFrame"] div {
        color: #7f1d1d !important;
    }

    /* Metric labels and values */
    [data-testid="stMetricLabel"] {
        color: #7f1d1d !important;
    }

    [data-testid="stMetricValue"] {
        color: #7f1d1d !important;
    }

    [data-testid="stMetricDelta"] {
        color: #7f1d1d !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2563eb;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white !important;
    }

    /* Info / success / warning / error messages */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div {
        color: #7f1d1d !important;
    }

    /* Selectbox input */
    input {
        color: #7f1d1d !important;
        background-color: white !important;
    }

    /* General Streamlit text */
    .stMarkdown,
    .stText,
    .stCaption {
        color: #7f1d1d !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# TITLE


st.markdown(
    '<h1>Insurance Claim Summary</h1>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="color:#0f172a;">
        View all submitted claims, their details, and current processing status.
    </p>
    """,
    unsafe_allow_html=True
)



# LOAD CLAIM STATUS


def load_claim_status():

    if not os.path.exists(CLAIM_STATUS_FILE):
        return []

    try:

        with open(
            CLAIM_STATUS_FILE,
            "rb"
        ) as f:

            data = pickle.load(f)

        if not isinstance(data, list):
            return []

        return data

    except Exception as e:

        st.error(
            f"Could not read {CLAIM_STATUS_FILE}: {e}"
        )

        return []



# LOAD CLAIM METADATA


def load_claim_metadata():

    if not os.path.exists(CLAIMS_METADATA_FILE):
        return []

    try:

        with open(
            CLAIMS_METADATA_FILE,
            "rb"
        ) as f:

            data = pickle.load(f)

        if not isinstance(data, list):
            return []

        return data

    except Exception as e:

        st.error(
            f"Could not read {CLAIMS_METADATA_FILE}: {e}"
        )

        return []



# CREATE CLAIM SUMMARY


def create_claim_summary(
    claim_metadata,
    claim_status
):

    
    # GROUP METADATA BY CLAIM NUMBER
    

    claims = {}

    for item in claim_metadata:

        claim_number = item.get(
            "claim_number"
        )

        if claim_number is None:
            continue

        claim_number = int(
            claim_number
        )

        if claim_number not in claims:

            claims[claim_number] = {
                "claim_number": claim_number,
                "claimant_name": item.get(
                    "claimant_name",
                    ""
                ),
                "claim_amount": item.get(
                    "claim_amount",
                    0.0
                ),
                "documents": set(),
                "document_chunks": 0
            }

       
        # DOCUMENT INFORMATION
       

        document_type = item.get(
            "document_type",
            ""
        )

        document_file = item.get(
            "document_file",
            ""
        )

        if document_type:
            claims[claim_number]["documents"].add(
                document_type
            )

        if document_file:
            claims[claim_number]["documents"].add(
                document_file
            )

        claims[claim_number]["document_chunks"] += 1


    
    # ADD STATUS INFORMATION
    

    status_lookup = {}

    for item in claim_status:

        claim_number = item.get(
            "claim_number"
        )

        if claim_number is None:
            continue

        status_lookup[
            int(claim_number)
        ] = item.get(
            "claim_status",
            "Unknown"
        )


    
    # BUILD FINAL SUMMARY
    

    summary = []

    for claim_number, claim in claims.items():

        status = status_lookup.get(
            claim_number,
            "Unknown"
        )

        summary.append(
            {
                "Claim Number":
                    claim_number,

                "Claimant Name":
                    claim["claimant_name"],

                "Claim Amount (Rs.)":
                    claim["claim_amount"],

                "Documents":
                    ", ".join(
                        sorted(
                            claim["documents"]
                        )
                    ),

                "Document Chunks":
                    claim["document_chunks"],

                "Claim Status":
                    status
            }
        )


    
    # INCLUDE STATUS RECORDS THAT MAY NOT YET HAVE METADATA
    

    existing_claim_numbers = set(
        claims.keys()
    )

    for item in claim_status:

        claim_number = item.get(
            "claim_number"
        )

        if claim_number is None:
            continue

        claim_number = int(
            claim_number
        )

        if claim_number not in existing_claim_numbers:

            summary.append(
                {
                    "Claim Number":
                        claim_number,

                    "Claimant Name":
                        "N/A",

                    "Claim Amount (Rs.)":
                        0.0,

                    "Documents":
                        "N/A",

                    "Document Chunks":
                        0,

                    "Claim Status":
                        item.get(
                            "claim_status",
                            "Unknown"
                        )
                }
            )


    
    # SORT BY CLAIM NUMBER
    

    summary.sort(
        key=lambda x: x["Claim Number"],
        reverse=True
    )

    return summary



# LOAD DATA


claim_status = load_claim_status()

claim_metadata = load_claim_metadata()



# CHECK WHETHER DATA EXISTS


if not claim_status and not claim_metadata:

    st.info(
        "No claims have been submitted yet."
    )

    st.stop()



# CREATE SUMMARY


claim_summary = create_claim_summary(
    claim_metadata,
    claim_status
)



# CLAIM COUNTS


total_claims = len(
    claim_summary
)

submitted_claims = sum(
    1
    for claim in claim_summary
    if claim["Claim Status"] == "Claim Submitted"
)

approved_claims = sum(
    1
    for claim in claim_summary
    if claim["Claim Status"] in [
        "Claim Approved",
        "Approved",
        "Auto Approved"
    ]
)

rejected_claims = sum(
    1
    for claim in claim_summary
    if claim["Claim Status"] in [
        "Claim Rejected",
        "Rejected"
    ]
)

human_review_claims = sum(
    1
    for claim in claim_summary
    if claim["Claim Status"] in [
        "Human Review",
        "Pending Human Approval",
        "Human Approval Required",
        "Escalated"
    ]
)



# SUMMARY METRICS


st.markdown(
    '<h3>Claim Overview</h3>',
    unsafe_allow_html=True
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Total Claims",
        total_claims
    )

with col2:

    st.metric(
        "Submitted",
        submitted_claims
    )

with col3:

    st.metric(
        "Approved",
        approved_claims
    )

with col4:

    st.metric(
        "Rejected",
        rejected_claims
    )

with col5:

    st.metric(
        "Human Review",
        human_review_claims
    )


st.divider()



# STATUS FILTER


st.markdown(
    '<h3>Filter Claims</h3>',
    unsafe_allow_html=True
)

all_statuses = sorted(
    set(
        claim["Claim Status"]
        for claim in claim_summary
    )
)

selected_status = st.selectbox(
    "Claim Status",
    ["All"] + all_statuses
)



# APPLY FILTER


filtered_claims = claim_summary

if selected_status != "All":

    filtered_claims = [
        claim
        for claim in claim_summary
        if claim["Claim Status"]
        == selected_status
    ]



# DISPLAY CLAIM TABLE


st.markdown(
    '<h3>All Claims</h3>',
    unsafe_allow_html=True
)


if not filtered_claims:

    st.info(
        "No claims found for the selected status."
    )

else:

    dataframe = pd.DataFrame(
        filtered_claims
    )


    # Format claim amount

    dataframe[
        "Claim Amount (Rs.)"
    ] = dataframe[
        "Claim Amount (Rs.)"
    ].apply(
        lambda x: f"Rs. {float(x):,.2f}"
    )


    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True
    )



# DETAILED CLAIM VIEW


st.divider()

st.markdown(
    '<h3>Claim Details</h3>',
    unsafe_allow_html=True
)


claim_numbers = [
    claim["Claim Number"]
    for claim in claim_summary
]


if claim_numbers:

    selected_claim = st.selectbox(
        "Select a claim to view details",
        claim_numbers
    )


    selected_data = next(
        (
            claim
            for claim in claim_summary
            if claim["Claim Number"]
            == selected_claim
        ),
        None
    )


    if selected_data:

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
                <p style="color:#7f1d1d;">
                    <b>Claim Number:</b>
                    {selected_data["Claim Number"]}
                </p>

                <p style="color:#7f1d1d;">
                    <b>Claimant Name:</b>
                    {selected_data["Claimant Name"]}
                </p>

                <p style="color:#7f1d1d;">
                    <b>Claim Amount:</b>
                    Rs. {float(selected_data["Claim Amount (Rs.)"]):,.2f}
                </p>
                """,
                unsafe_allow_html=True
            )

        with col2:

            status = selected_data[
                "Claim Status"
            ]

            if status in [
                "Claim Approved",
                "Approved",
                "Auto Approved"
            ]:

                st.success(
                    f"Status: {status}"
                )

            elif status in [
                "Claim Rejected",
                "Rejected"
            ]:

                st.error(
                    f"Status: {status}"
                )

            elif status in [
                "Human Review",
                "Pending Human Approval",
                "Human Approval Required",
                "Escalated"
            ]:

                st.warning(
                    f"Status: {status}"
                )

            else:

                st.info(
                    f"Status: {status}"
                )


        st.markdown(
            "<p style='color:#7f1d1d;'><b>Documents</b></p>",
            unsafe_allow_html=True
        )

        st.write(
            selected_data["Documents"]
        )

        st.markdown(
            "<p style='color:#7f1d1d;'><b>Document Chunks</b></p>",
            unsafe_allow_html=True
        )

        st.write(
            selected_data["Document Chunks"]
        )



# REFRESH BUTTON


st.divider()

if st.button("🔄 Refresh Claims"):

    st.rerun()
