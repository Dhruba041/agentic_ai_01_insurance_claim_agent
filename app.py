import streamlit as st
import os
import pickle
import json
import re
from typing import TypedDict, Optional

import numpy as np
import faiss

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

# STREAMLIT CONFIGURATION

st.set_page_config(
    page_title="AI Insurance Claim Processing",
    page_icon="🏦",
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

    h1, h2, h3 {
        color: #0f172a;
    }

    label {
        color: #0f172a !important;
        font-weight: 500;
    }

    input, textarea {
        background-color: white !important;
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

    .success-box {
        background-color: #dcfce7;
        border-left: 5px solid #16a34a;
        padding: 15px;
        border-radius: 8px;
        color: #14532d;
    }

    .warning-box {
        background-color: #fef3c7;
        border-left: 5px solid #d97706;
        padding: 15px;
        border-radius: 8px;
        color: #78350f;
    }

    .error-box {
        background-color: #fee2e2;
        border-left: 5px solid #dc2626;
        padding: 15px;
        border-radius: 8px;
        color: #7f1d1d;
    }

    .info-box {
        background-color: #dbeafe;
        border-left: 5px solid #2563eb;
        padding: 15px;
        border-radius: 8px;
        color: #1e3a8a;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# TITLE

st.markdown(
    '<h1 style="color:brown;">AI Insurance Claim Processing Agent</h1>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="color:#0f172a;">
    Enter a claim number to process the claim using document verification,
    eligibility checking, fraud detection, claim summarization and
    human approval where required.
    </p>
    """,
    unsafe_allow_html=True
)

# FILE LOCATIONS

POLICY_FAISS_FILE = "insurance_policy.faiss"
POLICY_METADATA_FILE = "insurance_policy.pkl"

CLAIMS_FAISS_FILE = "claims.faiss"
CLAIMS_METADATA_FILE = "claim_metadata.pkl"


# EMBEDDING MODEL

#@st.cache_resource
#def load_embedding_model():

#    return HuggingFaceEmbeddings(
#        model_name="BAAI/bge-small-en-v1.5"
#    )

#embedding_model = load_embedding_model()

embedding_model =  HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# LOCAL LLM

#@st.cache_resource
#def load_llm():

#    return ChatOllama(
#        model="gemma4:e2b",
#        temperature=0
#    )

#llm = load_llm()

llm = ChatOllama(
    model="gemma4:e2b",
    temperature=0
)

# LANGGRAPH CHECKPOINTER

@st.cache_resource
def load_checkpointer():

    return InMemorySaver()


checkpointer = load_checkpointer()

# JSON HELPER

def extract_json(text):

    """
    Attempts to extract JSON from the local LLM response.
    """

    if not text:
        return {}

    text = str(text).strip()

    # Remove markdown code fences

    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    text = text.strip()

    # First attempt

    try:

        return json.loads(text)

    except Exception:
        pass

    # Find JSON object

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if match:

        try:

            return json.loads(
                match.group(0)
            )

        except Exception:
            pass

    return {}


# LLM HELPER

def ask_llm(prompt):

    response = llm.invoke(prompt)

    return response.content


# LOAD POLICY

def load_policy():

    if not os.path.exists(
        POLICY_FAISS_FILE
    ):

        return None, []

    if not os.path.exists(
        POLICY_METADATA_FILE
    ):

        return None, []

    try:

        policy_index = faiss.read_index(
            POLICY_FAISS_FILE
        )

        with open(
            POLICY_METADATA_FILE,
            "rb"
        ) as f:

            policy_metadata = pickle.load(
                f
            )

        return (
            policy_index,
            policy_metadata
        )

    except Exception:

        return None, []


# LOAD CLAIM DATA

def load_claim_data(claim_number):

    if not os.path.exists(
        CLAIMS_METADATA_FILE
    ):

        return []

    try:

        with open(
            CLAIMS_METADATA_FILE,
            "rb"
        ) as f:

            metadata = pickle.load(
                f
            )

    except Exception:

        return []

    claim_number = int(
        claim_number
    )

    return [
        item
        for item in metadata
        if int(item["claim_number"]) == claim_number
    ]


# RETRIEVE POLICY CONTEXT

def retrieve_policy_context(
    query,
    top_k=8
):

    policy_index, policy_metadata = load_policy()

    if policy_index is None:

        return "Insurance policy could not be loaded."

    try:

        query_embedding = embedding_model.embed_query(
            query
        )

        query_embedding = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        faiss.normalize_L2(
            query_embedding
        )

        k = min(
            top_k,
            policy_index.ntotal
        )

        if k == 0:

            return "No policy information available."

        distances, indices = policy_index.search(
            query_embedding,
            k
        )

        results = []

        for index_position in indices[0]:

            if index_position < 0:
                continue

            if index_position >= len(policy_metadata):
                continue

            results.append(
                policy_metadata[index_position]["text"]
            )

        return "\n\n".join(
            results
        )

    except Exception as e:

        return (
            f"Unable to retrieve policy context: {e}"
        )


# GET CLAIM DOCUMENT TEXT

def get_claim_document_text(
    claim_records,
    document_type
):

    records = [
        record
        for record in claim_records
        if record["document_type"] == document_type
    ]

    records.sort(
        key=lambda x: x["chunk_id"]
    )

    return "\n".join(
        record["text"]
        for record in records
    )


# LANGGRAPH STATE

class ClaimState(TypedDict, total=False):

    claim_number: int

    claimant_name: str

    claim_amount: float

    identification_text: str

    bill_text: str

    policy_context: str

    document_result: dict

    eligibility_result: dict

    fraud_result: dict

    claim_summary: str

    decision: str

    decision_reason: str

    human_decision: str

    processing_status: str

# NODE 1
# DOCUMENT VERIFICATION AGENT

def document_verification_agent(
    state: ClaimState
):

    identification_text = state.get(
        "identification_text",
        ""
    )

    bill_text = state.get(
        "bill_text",
        ""
    )

    claimant_name = state.get(
        "claimant_name",
        ""
    )

    # Deterministic document presence check

    missing_documents = []

    if not identification_text.strip():

        missing_documents.append(
            "Identification proof"
        )

    if not bill_text.strip():

        missing_documents.append(
            "Bill/supporting document"
        )

    # If documents are missing, don't rely
    # on LLM to say that they exist.

    if missing_documents:

        return {
            "document_result": {
                "status": "MISSING",
                "identification_valid": bool(
                    identification_text.strip()
                ),
                "bill_valid": bool(
                    bill_text.strip()
                ),
                "missing_documents":
                    missing_documents,
                "reason":
                    "Required claim documents are missing."
            }
        }

    policy_context = state.get(
        "policy_context",
        ""
    )

    prompt = f"""
You are the Document Verification Agent for an insurance company.

Verify the submitted identification proof and bill.

Insurance policy:
{policy_context}

Claimant name:
{claimant_name}

Identification document:
{identification_text}

Bill/supporting document:
{bill_text}

Check:

1. Whether the identification document appears to be
   one of the permitted government IDs.
2. Whether the name reasonably matches the claimant.
3. Whether the bill contains:
   - service provider
   - date
   - description
   - amount
   - bill/invoice number
4. Whether the bill appears readable and internally consistent.
5. Whether any document appears altered or suspicious.

Do NOT declare fraud merely because something is unusual.

Return ONLY JSON:

{{
    "identification_valid": true,
    "bill_valid": true,
    "name_match": true,
    "bill_details_complete": true,
    "document_irregularity": false,
    "missing_documents": [],
    "reason": "short explanation"
}}
"""

    response = ask_llm(
        prompt
    )

    result = extract_json(
        response
    )

    if not result:

        result = {
            "identification_valid": False,
            "bill_valid": False,
            "name_match": False,
            "bill_details_complete": False,
            "document_irregularity": False,
            "missing_documents": [],
            "reason":
                "The document verification model did not return a valid result."
        }

    # Ensure missing documents are deterministic

    result["status"] = (
        "VALID"
        if (
            result.get(
                "identification_valid",
                False
            )
            and
            result.get(
                "bill_valid",
                False
            )
            and
            result.get(
                "name_match",
                False
            )
            and
            result.get(
                "bill_details_complete",
                False
            )
        )
        else "INVALID"
    )

    return {
        "document_result": result
    }



# NODE 2
# ELIGIBILITY CHECK AGENT


def eligibility_check_agent(
    state: ClaimState
):

    policy_context = state.get(
        "policy_context",
        ""
    )

    identification_text = state.get(
        "identification_text",
        ""
    )

    bill_text = state.get(
        "bill_text",
        ""
    )

    claim_amount = state.get(
        "claim_amount",
        0
    )

    prompt = f"""
You are the Eligibility Check Agent for an insurance company.

Determine whether the claim satisfies the insurance policy.

Insurance policy:
{policy_context}

Claimant:
{state.get("claimant_name", "")}

Claim amount:
Rs. {claim_amount}

Identification document:
{identification_text}

Bill/supporting document:
{bill_text}

Eligibility requirements:

1. Policy must be active on the incident date.
2. Claimant must be the policyholder, insured person,
   nominee, or authorized claimant.
3. Incident must be covered.
4. Incident must occur during policy period.
5. Claim must be reported within applicable notification period,
   unless a valid delay is accepted.
6. Loss must have sufficient evidence.
7. Loss must not fall under an exclusion.
8. Claim must be within applicable coverage limits after
   deductibles and adjustments.

Important:
Do not invent policy dates or coverage information.
If required information is unavailable or unclear,
mark the result as UNCLEAR rather than assuming eligibility.

Return ONLY JSON:

{{
    "status": "ELIGIBLE",
    "policy_active": true,
    "claimant_eligible": true,
    "incident_covered": true,
    "incident_within_policy_period": true,
    "notification_compliant": true,
    "loss_supported": true,
    "exclusion_applies": false,
    "within_coverage_limit": true,
    "reason": "short explanation"
}}

Allowed status values:
ELIGIBLE
INELIGIBLE
UNCLEAR
"""

    response = ask_llm(
        prompt
    )

    result = extract_json(
        response
    )

    if not result:

        result = {
            "status": "UNCLEAR",
            "reason":
                "Eligibility could not be determined."
        }

    return {
        "eligibility_result": result
    }



# NODE 3
# FRAUD DETECTION AGENT


def fraud_detection_agent(
    state: ClaimState
):

    policy_context = state.get(
        "policy_context",
        ""
    )

    identification_text = state.get(
        "identification_text",
        ""
    )

    bill_text = state.get(
        "bill_text",
        ""
    )

    claim_amount = state.get(
        "claim_amount",
        0
    )

    prompt = f"""
You are the Fraud Detection Agent for an insurance company.

Analyze the claim for potential fraud indicators.

Insurance policy:
{policy_context}

Claim amount:
Rs. {claim_amount}

Identification:
{identification_text}

Bill:
{bill_text}

Potential indicators include:

- altered or forged documents
- duplicate bills or claims
- inconsistent dates
- inconsistent names
- inconsistent amounts
- inconsistent incident details
- unverifiable service providers
- inflated or unreasonable claim amount
- loss before policy activation
- repeated claims
- false or misleading information
- evidence that the loss did not occur
- major discrepancies between documents

Important:
A fraud indicator DOES NOT establish fraud.

A high claim amount alone is NOT fraud.

Claims of Rs. 10,000 or more simply require human approval
according to the company policy.

Return ONLY JSON:

{{
    "fraud_indicator": false,
    "risk_level": "LOW",
    "indicators": [],
    "amount_concern": false,
    "document_concern": false,
    "reason": "short explanation"
}}

Allowed risk levels:
LOW
MEDIUM
HIGH
"""

    response = ask_llm(
        prompt
    )

    result = extract_json(
        response
    )

    if not result:

        result = {
            "fraud_indicator": False,
            "risk_level": "MEDIUM",
            "indicators": [],
            "amount_concern": False,
            "document_concern": False,
            "reason":
                "Fraud analysis could not be completed."
        }

    return {
        "fraud_result": result
    }



# NODE 4
# CLAIM SUMMARY AGENT


def claim_summary_agent(
    state: ClaimState
):

    document_result = state.get(
        "document_result",
        {}
    )

    eligibility_result = state.get(
        "eligibility_result",
        {}
    )

    fraud_result = state.get(
        "fraud_result",
        {}
    )

    summary = f"""
Claim Number: {state.get("claim_number")}

Claimant Name:
{state.get("claimant_name")}

Claim Amount:
Rs. {state.get("claim_amount", 0):,.2f}

DOCUMENT VERIFICATION
Status: {document_result.get("status")}
Reason: {document_result.get("reason")}

ELIGIBILITY
Status: {eligibility_result.get("status")}
Reason: {eligibility_result.get("reason")}

FRAUD ASSESSMENT
Risk Level: {fraud_result.get("risk_level")}
Fraud Indicator: {fraud_result.get("fraud_indicator")}
Indicators: {fraud_result.get("indicators")}
Reason: {fraud_result.get("reason")}
"""

    return {
        "claim_summary": summary
    }


# NODE 5
# DECISION / ROUTING LOGIC

def determine_decision(
    state: ClaimState
):

    document_result = state.get(
        "document_result",
        {}
    )

    eligibility_result = state.get(
        "eligibility_result",
        {}
    )

    fraud_result = state.get(
        "fraud_result",
        {}
    )

    claim_amount = float(
        state.get(
            "claim_amount",
            0
        )
    )

    document_status = document_result.get(
        "status"
    )

    eligibility_status = eligibility_result.get(
        "status"
    )

    fraud_indicator = fraud_result.get(
        "fraud_indicator",
        False
    )

    fraud_risk = fraud_result.get(
        "risk_level",
        "LOW"
    )

    
    # REJECTION CONDITIONS
    

    if eligibility_status == "INELIGIBLE":

        return {
            "decision": "REJECT",
            "decision_reason":
                "Claim does not satisfy the policy eligibility requirements.",
            "processing_status": "COMPLETED"
        }

    # Missing documents are a policy failure in this workflow.

    if document_status == "MISSING":

        return {
            "decision": "REJECT",
            "decision_reason":
                "Required claim documents are missing.",
            "processing_status": "COMPLETED"
        }

    # Clearly invalid documents

    if document_status == "INVALID":

        return {
            "decision": "REJECT",
            "decision_reason":
                "Required identification or supporting documents "
                "could not be verified.",
            "processing_status": "COMPLETED"
        }

    
    # HUMAN REVIEW CONDITIONS
    

    if eligibility_status == "UNCLEAR":

        return {
            "decision": "HUMAN_APPROVAL",
            "decision_reason":
                "Claim eligibility could not be conclusively determined.",
            "processing_status": "PENDING_HUMAN"
        }

    if fraud_indicator:

        return {
            "decision": "HUMAN_APPROVAL",
            "decision_reason":
                "Potential fraud or irregularity indicators require "
                "human investigation.",
            "processing_status": "PENDING_HUMAN"
        }

    if fraud_risk in ["MEDIUM", "HIGH"]:

        return {
            "decision": "HUMAN_APPROVAL",
            "decision_reason":
                "The fraud detection agent identified elevated risk.",
            "processing_status": "PENDING_HUMAN"
        }

    
    # CLAIM AMOUNT RULE
    

    # Policy:
    # Below Rs. 10,000 can be auto-approved.
    # Rs. 10,000 or more requires human approval.

    if claim_amount >= 10000:

        return {
            "decision": "HUMAN_APPROVAL",
            "decision_reason":
                "Claim amount is Rs. 10,000 or more and therefore "
                "requires mandatory human approval.",
            "processing_status": "PENDING_HUMAN"
        }

    
    # AUTO APPROVAL
    

    return {
        "decision": "AUTO_APPROVE",
        "decision_reason":
            "Claim is eligible, required documents are valid, "
            "no significant fraud indicators were identified, "
            "and the claim amount is below Rs. 10,000.",
        "processing_status": "COMPLETED"
    }



# NODE 6
# HUMAN APPROVAL AGENT


def human_approval_agent(
    state: ClaimState
):

    # LangGraph interrupts execution here.
    # The graph resumes using Command(resume=...).

    approval_request = {
        "claim_number":
            state.get("claim_number"),

        "claimant_name":
            state.get("claimant_name"),

        "claim_amount":
            state.get("claim_amount"),

        "reason":
            state.get("decision_reason"),

        "fraud_result":
            state.get("fraud_result"),

        "eligibility_result":
            state.get("eligibility_result"),

        "document_result":
            state.get("document_result"),

        "summary":
            state.get("claim_summary"),

        "message":
            "Human approval is required. "
            "Select APPROVE, PARTIAL APPROVE or REJECT."
    }

    human_decision = interrupt(
        approval_request
    )

    if isinstance(
        human_decision,
        dict
    ):

        decision = human_decision.get(
            "decision",
            "REJECT"
        )

    else:

        decision = str(
            human_decision
        ).upper()

    if decision not in [
        "APPROVE",
        "PARTIAL APPROVE",
        "REJECT"
    ]:

        decision = "REJECT"

    return {
        "human_decision": decision,
        "decision": decision,
        "processing_status": "COMPLETED",
        "decision_reason":
            f"Human claims officer decision: {decision}"
    }



# CONDITIONAL ROUTING


def route_after_decision(
    state: ClaimState
):

    decision = state.get(
        "decision"
    )

    if decision == "AUTO_APPROVE":

        return "END"

    if decision == "REJECT":

        return "END"

    if decision == "HUMAN_APPROVAL":

        return "human_approval"

    return "END"



# BUILD LANGGRAPH


@st.cache_resource
def build_graph():

    builder = StateGraph(
        ClaimState
    )

    
    # ADD NODES
    

    builder.add_node(
        "document_verification",
        document_verification_agent
    )

    builder.add_node(
        "eligibility_check",
        eligibility_check_agent
    )

    builder.add_node(
        "fraud_detection",
        fraud_detection_agent
    )

    builder.add_node(
        "claim_summary",
        claim_summary_agent
    )

    builder.add_node(
        "determine_decision",
        determine_decision
    )

    builder.add_node(
        "human_approval",
        human_approval_agent
    )

    
    # PARALLEL EXECUTION
    

    # START fans out to three independent agents.

    builder.add_edge(
        START,
        "document_verification"
    )

    builder.add_edge(
        START,
        "eligibility_check"
    )

    builder.add_edge(
        START,
        "fraud_detection"
    )

    # All three converge at summary.

    builder.add_edge(
        "document_verification",
        "claim_summary"
    )

    builder.add_edge(
        "eligibility_check",
        "claim_summary"
    )

    builder.add_edge(
        "fraud_detection",
        "claim_summary"
    )

    # Summary -> decision

    builder.add_edge(
        "claim_summary",
        "determine_decision"
    )

    # Conditional routing

    builder.add_conditional_edges(
        "determine_decision",
        route_after_decision,
        {
            "human_approval":
                "human_approval",

            "END":
                END
        }
    )

    # Human approval -> END

    builder.add_edge(
        "human_approval",
        END
    )

    return builder.compile(
        checkpointer=checkpointer
    )


graph = build_graph()



# CLAIM PROCESSING FUNCTION


def process_claim(
    claim_number
):

    claim_records = load_claim_data(
        claim_number
    )

    if not claim_records:

        return {
            "error":
                f"Claim number {claim_number} was not found."
        }

    
    # GET CLAIM DETAILS
    

    first_record = claim_records[0]

    claimant_name = first_record.get(
        "claimant_name",
        ""
    )

    claim_amount = float(
        first_record.get(
            "claim_amount",
            0
        )
    )

    identification_text = get_claim_document_text(
        claim_records,
        "identification"
    )

    bill_text = get_claim_document_text(
        claim_records,
        "bill"
    )

    
    # POLICY RAG
    

    policy_query = f"""
Insurance claim processing policy for:
claim eligibility,
required documents,
fraud detection,
automatic approval,
human approval,
rejection,
claim amount Rs. {claim_amount}
"""

    policy_context = retrieve_policy_context(
        policy_query
    )

    
    # GRAPH INPUT
    

    initial_state = {

        "claim_number":
            int(claim_number),

        "claimant_name":
            claimant_name,

        "claim_amount":
            claim_amount,

        "identification_text":
            identification_text,

        "bill_text":
            bill_text,

        "policy_context":
            policy_context
    }

    # Use claim number as stable thread ID.

    config = {
        "configurable": {
            "thread_id":
                f"claim-{claim_number}"
        }
    }

    result = graph.invoke(
        initial_state,
        config=config
    )

    return result



# UI - CLAIM NUMBER


st.markdown(
    '<h3 style="color:brown;">Process Claim</h3>',
    unsafe_allow_html=True
)

claim_number_input = st.number_input(
    "Enter Claim Number",
    min_value=1000,
    step=1,
    value=1000
)



# START PROCESSING


if st.button(
    "Process Claim"
):

    claim_number = int(
        claim_number_input
    )

    with st.spinner(
        "AI agents are processing the claim..."
    ):

        result = process_claim(
            claim_number
        )

    if result.get("error"):

        st.error(
            result["error"]
        )

        st.stop()

    
    # CHECK FOR HUMAN INTERRUPT
    

    if "__interrupt__" in result:

        interrupt_data = result[
            "__interrupt__"
        ]

        if interrupt_data:

            interrupt_value = interrupt_data[
                0
            ].value

            st.session_state[
                "pending_claim"
            ] = claim_number

            st.session_state[
                "pending_interrupt"
            ] = interrupt_value

            st.session_state[
                "claim_result"
            ] = result

            st.rerun()


    
    # DISPLAY RESULT
    

    st.session_state[
        "claim_result"
    ] = result


# HUMAN APPROVAL SECTION

if (
    "pending_interrupt"
    in st.session_state
):

    approval_data = st.session_state[
        "pending_interrupt"
    ]

    st.markdown(
        '<h2 style="color:brown;">Human Approval Required</h2>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="warning-box">'
        '<b>This claim cannot be automatically settled.</b>'
        '<br><br>'
        'A claims officer must review the claim.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### Claim Details"
    )

    st.write(
        "Claim Number:",
        approval_data.get(
            "claim_number"
        )
    )

    st.write(
        "Claimant:",
        approval_data.get(
            "claimant_name"
        )
    )

    st.write(
        "Claim Amount:",
        f"Rs. {approval_data.get('claim_amount', 0):,.2f}"
    )

    st.write(
        "Reason:",
        approval_data.get(
            "reason"
        )
    )

    st.markdown(
        "### AI Claim Summary"
    )

    st.text(
        approval_data.get(
            "summary",
            ""
        )
    )

    st.markdown(
        "### Fraud Assessment"
    )

    st.json(
        approval_data.get(
            "fraud_result",
            {}
        )
    )

    st.markdown(
        "### Eligibility Assessment"
    )

    st.json(
        approval_data.get(
            "eligibility_result",
            {}
        )
    )

    human_decision = st.selectbox(
        "Claims Officer Decision",
        [
            "APPROVE",
            "PARTIAL APPROVE",
            "REJECT"
        ]
    )

    if st.button(
        "Submit Human Decision"
    ):

        claim_number = st.session_state[
            "pending_claim"
        ]

        config = {
            "configurable": {
                "thread_id":
                    f"claim-{claim_number}"
            }
        }

        with st.spinner(
            "Updating claim decision..."
        ):

            final_result = graph.invoke(
                Command(
                    resume={
                        "decision":
                            human_decision
                    }
                ),
                config=config
            )

        # Clear pending state

        del st.session_state[
            "pending_claim"
        ]

        del st.session_state[
            "pending_interrupt"
        ]

        st.session_state[
            "claim_result"
        ] = final_result

        st.rerun()



# DISPLAY FINAL RESULT


if (
    "claim_result"
    in st.session_state
):

    result = st.session_state[
        "claim_result"
    ]

    if (
        "__interrupt__"
        not in result
    ):

        decision = result.get(
            "decision"
        )

        st.markdown(
            "---"
        )

        st.markdown(
            '<h2 style="color:brown;">Claim Processing Result</h2>',
            unsafe_allow_html=True
        )

        st.write(
            "Claim Number:",
            result.get(
                "claim_number"
            )
        )

        st.write(
            "Claimant:",
            result.get(
                "claimant_name"
            )
        )

        st.write(
            "Claim Amount:",
            f"Rs. {result.get('claim_amount', 0):,.2f}"
        )

        
        # DECISION DISPLAY
        

        if decision == "AUTO_APPROVE":

            st.markdown(
                """
                <div class="success-box">
                <h3>CLAIM AUTO APPROVED</h3>
                The claim satisfies the automatic approval
                conditions under the insurance policy.
                </div>
                """,
                unsafe_allow_html=True
            )

        elif decision == "REJECT":

            st.markdown(
                """
                <div class="error-box">
                <h3>CLAIM REJECTED</h3>
                The claim does not satisfy the applicable
                policy requirements.
                </div>
                """,
                unsafe_allow_html=True
            )

        elif decision in [
            "APPROVE",
            "PARTIAL APPROVE"
        ]:

            st.markdown(
                f"""
                <div class="success-box">
                <h3>HUMAN DECISION: {decision}</h3>
                The claim was reviewed by an authorized
                claims officer.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="warning-box">
                <h3>HUMAN REVIEW REQUIRED</h3>
                The claim requires manual assessment.
                </div>
                """,
                unsafe_allow_html=True
            )

        # REASON

        st.markdown(
            "### Decision Reason"
        )

        st.write(
            result.get(
                "decision_reason",
                ""
            )
        )

        # SUMMARY
        
        st.markdown(
            "### Claim Summary"
        )

        st.text(
            result.get(
                "claim_summary",
                ""
            )
        )

        # AGENT RESULTS
        

        with st.expander(
            "Document Verification Result"
        ):

            st.json(
                result.get(
                    "document_result",
                    {}
                )
            )

        with st.expander(
            "Eligibility Result"
        ):

            st.json(
                result.get(
                    "eligibility_result",
                    {}
                )
            )

        with st.expander(
            "Fraud Detection Result"
        ):

            st.json(
                result.get(
                    "fraud_result",
                    {}
                )
            )
