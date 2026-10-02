
# STREAMLIT CLAIM PROCESSING APPLICATION


import streamlit as st
import pickle
import os

from claim_processor import (
    process_claim,
    finalize_human_decision,
    get_claim_documents,
    load_claim_status
)



# STREAMLIT CONFIGURATION


st.set_page_config(
    page_title="Insurance Claim Processing",
    page_icon="🏥",
    layout="wide"
)



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

    h1 {
        color: #0f172a;
    }

    h2 {
        color: #0f172a;
    }

    h3 {
        color: #7f1d1d;
    }

    label {
        color: #0f172a !important;
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

    [data-testid="stDataFrame"] {
        background-color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# TITLE


st.markdown(
    """
    <h1 style="color:brown;">
        Insurance Claim Processing
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="color:#0f172a;">
        Select a submitted claim and run the AI-powered
        LangGraph claim processing workflow.
    </p>
    """,
    unsafe_allow_html=True
)



# CONSTANTS


CLAIM_STATUS_FILE = "claim_status.pkl"



# LOAD CLAIM METADATA


def load_claim_metadata():

    metadata_file = "claim_metadata.pkl"

    if not os.path.exists(
        metadata_file
    ):

        return []

    try:

        with open(
            metadata_file,
            "rb"
        ) as f:

            metadata = pickle.load(
                f
            )

            if isinstance(
                metadata,
                list
            ):

                return metadata

            return []

    except Exception:

        return []



# GET CLAIM DETAILS


def get_claim_basic_details(
    claim_number
):

    metadata = load_claim_metadata()

    claimant_name = ""

    claim_amount = 0.0

    document_types = set()

    for item in metadata:

        if int(
            item.get(
                "claim_number",
                -1
            )
        ) == int(
            claim_number
        ):

            claimant_name = item.get(
                "claimant_name",
                claimant_name
            )

            claim_amount = float(
                item.get(
                    "claim_amount",
                    claim_amount
                )
            )

            document_types.add(
                item.get(
                    "document_type",
                    ""
                )
            )

    return (
        claimant_name,
        claim_amount,
        document_types
    )



# LOAD CLAIM STATUS


claim_status_data = load_claim_status()



# FILTER PENDING CLAIMS


pending_claims = [

    item

    for item in claim_status_data

    if item.get(
        "claim_status"
    ) == "Claim Submitted"

]



# DISPLAY PENDING CLAIMS


st.markdown(
    """
    <h2 style="color:brown;">
        Claims Pending Processing
    </h2>
    """,
    unsafe_allow_html=True
)


if not pending_claims:

    st.info(
        "There are currently no claims pending processing."
    )

    st.stop()



# CREATE DISPLAY DATA


display_data = []

for item in pending_claims:

    claim_number = int(
        item["claim_number"]
    )

    claimant_name, claim_amount, document_types = (
        get_claim_basic_details(
            claim_number
        )
    )

    display_data.append(
        {
            "Claim Number":
                claim_number,

            "Claimant Name":
                claimant_name,

            "Claim Amount":
                f"Rs. {claim_amount:,.2f}",

            "Status":
                item.get(
                    "claim_status",
                    ""
                )
        }
    )



# DISPLAY TABLE


st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)



# CLAIM SELECTION


st.markdown(
    """
    <h3>
        Select Claim to Process
    </h3>
    """,
    unsafe_allow_html=True
)


claim_numbers = [

    int(
        item["claim_number"]
    )

    for item in pending_claims

]


selected_claim = st.selectbox(
    "Select Claim Number",
    claim_numbers
)



# SELECTED CLAIM DETAILS


claimant_name, claim_amount, document_types = (
    get_claim_basic_details(
        selected_claim
    )
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Claim Number",
        selected_claim
    )


with col2:

    st.metric(
        "Claimant",
        claimant_name
    )


with col3:

    st.metric(
        "Claim Amount",
        f"Rs. {claim_amount:,.2f}"
    )



# DOCUMENT TYPES


st.write(
    "### Submitted Documents"
)

for document_type in sorted(
    document_types
):

    st.write(
        f"✓ {document_type.title()}"
    )



# PROCESS CLAIM BUTTON


st.markdown("---")


if st.button(
    "Process Selected Claim",
    type="primary"
):

    with st.spinner(
        f"Processing Claim {selected_claim}..."
    ):

        try:

            result = process_claim(
                selected_claim
            )

            st.session_state[
                "claim_result"
            ] = result

            st.session_state[
                "processed_claim_number"
            ] = selected_claim

            st.success(
                "Claim processing completed."
            )

        except Exception as e:

            st.error(
                f"Claim processing failed: {e}"
            )

            st.stop()



# DISPLAY PROCESSING RESULT


if (
    "claim_result"
    in st.session_state
):

    result = st.session_state[
        "claim_result"
    ]

    processed_claim_number = (
        st.session_state[
            "processed_claim_number"
        ]
    )

    
    # Only display result for currently selected claim
    

    if (
        processed_claim_number
        == selected_claim
    ):

        st.markdown("---")

        st.markdown(
            """
            <h2 style="color:brown;">
                AI Claim Assessment
            </h2>
            """,
            unsafe_allow_html=True
        )


        
        # DOCUMENT RESULT
        

        document_result = result.get(
            "document_result",
            {}
        )

        st.markdown(
            "### 1. Document Verification"
        )

        doc_col1, doc_col2, doc_col3 = (
            st.columns(3)
        )


        with doc_col1:

            st.write(
                "Documents Complete"
            )

            if document_result.get(
                "documents_complete"
            ):

                st.success("Yes")

            else:

                st.error("No")


        with doc_col2:

            st.write(
                "Identification"
            )

            if document_result.get(
                "identification_valid"
            ):

                st.success("Valid")

            else:

                st.error("Invalid")


        with doc_col3:

            st.write(
                "Bill"
            )

            if document_result.get(
                "bill_valid"
            ):

                st.success("Valid")

            else:

                st.error("Invalid")


        missing_documents = (
            document_result.get(
                "missing_documents",
                []
            )
        )


        if missing_documents:

            st.warning(
                "Missing Documents: "
                + ", ".join(
                    missing_documents
                )
            )


        document_issues = (
            document_result.get(
                "document_issues",
                []
            )
        )


        if document_issues:

            st.warning(
                "Document Issues: "
                + "; ".join(
                    document_issues
                )
            )


        
        # ELIGIBILITY
        

        eligibility_result = result.get(
            "eligibility_result",
            {}
        )

        st.markdown(
            "### 2. Eligibility Check"
        )


        eligibility = (
            eligibility_result.get(
                "eligibility",
                "unknown"
            )
        )


        if eligibility == "eligible":

            st.success(
                "Claim appears eligible."
            )

        elif eligibility == "ineligible":

            st.error(
                "Claim appears ineligible."
            )

        else:

            st.warning(
                "Eligibility could not be established."
            )


        st.write(
            eligibility_result.get(
                "reason",
                ""
            )
        )


        eligibility_issues = (
            eligibility_result.get(
                "eligibility_issues",
                []
            )
        )


        if eligibility_issues:

            st.write(
                "**Eligibility Issues:**"
            )

            for issue in eligibility_issues:

                st.write(
                    f"- {issue}"
                )


        
        # FRAUD
        

        fraud_result = result.get(
            "fraud_result",
            {}
        )

        st.markdown(
            "### 3. Fraud / Irregularity Check"
        )


        fraud_risk = fraud_result.get(
            "fraud_risk",
            "unknown"
        )


        if fraud_risk == "low":

            st.success(
                "Fraud risk: Low"
            )

        elif fraud_risk == "medium":

            st.warning(
                "Fraud risk: Medium"
            )

        elif fraud_risk == "high":

            st.error(
                "Fraud risk: High"
            )

        else:

            st.info(
                f"Fraud risk: {fraud_risk}"
            )


        st.write(
            fraud_result.get(
                "reason",
                ""
            )
        )


        fraud_indicators = (
            fraud_result.get(
                "fraud_indicators",
                []
            )
        )


        if fraud_indicators:

            st.write(
                "**Potential Indicators:**"
            )

            for indicator in fraud_indicators:

                st.write(
                    f"- {indicator}"
                )


        
        # CLAIM SUMMARY
        

        summary_result = result.get(
            "summary_result",
            {}
        )

        st.markdown(
            "### 4. Claim Summary"
        )


        st.write(
            summary_result.get(
                "claim_overview",
                ""
            )
        )


        with st.expander(
            "Detailed Assessment"
        ):

            st.write(
                "**Document Assessment**"
            )

            st.write(
                summary_result.get(
                    "document_assessment",
                    ""
                )
            )


            st.write(
                "**Eligibility Assessment**"
            )

            st.write(
                summary_result.get(
                    "eligibility_assessment",
                    ""
                )
            )


            st.write(
                "**Fraud Assessment**"
            )

            st.write(
                summary_result.get(
                    "fraud_assessment",
                    ""
                )
            )


            important_issues = (
                summary_result.get(
                    "important_issues",
                    []
                )
            )


            if important_issues:

                st.write(
                    "**Important Issues**"
                )

                for issue in important_issues:

                    st.write(
                        f"- {issue}"
                    )


        
        # FINAL RESULT
        

        processing_status = result.get(
            "processing_status",
            ""
        )


        st.markdown("---")

        st.markdown(
            "### 5. Processing Decision"
        )


        
        # AUTOMATIC APPROVAL
        

        if processing_status == "Approved":

            st.success(
                "CLAIM AUTOMATICALLY APPROVED"
            )

            st.write(
                result.get(
                    "final_reason",
                    ""
                )
            )


        
        # AUTOMATIC REJECTION
        

        elif processing_status == "Rejected":

            st.error(
                "CLAIM REJECTED"
            )

            st.write(
                result.get(
                    "final_reason",
                    ""
                )
            )


        
        # HUMAN APPROVAL
        

        elif (
            processing_status
            == "Human Approval Required"
        ):

            st.warning(
                "HUMAN APPROVAL REQUIRED"
            )


            human_result = result.get(
                "human_result",
                {}
            )


            st.write(
                human_result.get(
                    "message",
                    "Claims officer review is required."
                )
            )


            human_reasons = (
                human_result.get(
                    "reasons",
                    []
                )
            )


            if human_reasons:

                st.write(
                    "**Reasons for Human Review:**"
                )

                for reason in human_reasons:

                    st.write(
                        f"- {reason}"
                    )


            st.markdown(
                "### Claims Officer Decision"
            )


            human_reason = st.text_area(
                "Decision Reason",
                placeholder=(
                    "Enter the reason for approving "
                    "or rejecting this claim."
                ),
                key=f"human_reason_{selected_claim}"
            )


            approve_col, reject_col = (
                st.columns(2)
            )


            with approve_col:

                if st.button(
                    "Approve Claim",
                    key=f"approve_{selected_claim}"
                ):

                    if not human_reason.strip():

                        st.error(
                            "Please enter a decision reason."
                        )

                    else:

                        try:

                            final_state = (
                                finalize_human_decision(
                                    selected_claim,
                                    "approve",
                                    human_reason
                                )
                            )

                            st.session_state[
                                "claim_result"
                            ] = final_state

                            st.success(
                                f"Claim {selected_claim} "
                                "has been approved."
                            )

                            st.rerun()

                        except Exception as e:

                            st.error(
                                f"Could not approve claim: {e}"
                            )


            with reject_col:

                if st.button(
                    "Reject Claim",
                    key=f"reject_{selected_claim}"
                ):

                    if not human_reason.strip():

                        st.error(
                            "Please enter a decision reason."
                        )

                    else:

                        try:

                            final_state = (
                                finalize_human_decision(
                                    selected_claim,
                                    "reject",
                                    human_reason
                                )
                            )

                            st.session_state[
                                "claim_result"
                            ] = final_state

                            st.error(
                                f"Claim {selected_claim} "
                                "has been rejected."
                            )

                            st.rerun()

                        except Exception as e:

                            st.error(
                                f"Could not reject claim: {e}"
                            )


        else:

            st.info(
                f"Processing Status: {processing_status}"
            )
