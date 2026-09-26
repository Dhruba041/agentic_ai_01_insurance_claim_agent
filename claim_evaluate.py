
# CLAIM PROCESSOR - LANGGRAPH + LOCAL OLLAMA LLM


import os
import pickle
from typing import TypedDict, Optional, Dict, Any, List

import faiss
import numpy as np

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
"""
from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.types import Send



# FILE LOCATIONS


CLAIMS_FAISS_FILE = "claims.faiss"

CLAIMS_METADATA_FILE = "claim_metadata.pkl"

CLAIM_STATUS_FILE = "claim_status.pkl"

POLICY_FAISS_FILE = "insurance_policy.faiss"

POLICY_METADATA_FILE = "insurance_policy.pkl"

CLAIM_RESULTS_FILE = "claim_results.pkl"



# LLM


llm = ChatOllama(
    model="gemma4:e2b",
    temperature=0
)



# EMBEDDING MODEL


embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)



# GRAPH STATE


class ClaimState(TypedDict, total=False):

    claim_number: int

    claimant_name: str

    claim_amount: float

    claim_documents: List[Dict[str, Any]]

    policy_context: str

    document_result: Dict[str, Any]

    eligibility_result: Dict[str, Any]

    fraud_result: Dict[str, Any]

    summary_result: Dict[str, Any]

    human_result: Dict[str, Any]

    final_decision: str

    final_reason: str

    processing_status: str



# CLAIM STATUS FUNCTIONS


def load_claim_status():

    if not os.path.exists(CLAIM_STATUS_FILE):
        return []

    try:

        with open(
            CLAIM_STATUS_FILE,
            "rb"
        ) as f:

            data = pickle.load(f)

            if isinstance(data, list):
                return data

            return []

    except Exception:

        return []


def save_claim_status_data(data):

    with open(
        CLAIM_STATUS_FILE,
        "wb"
    ) as f:

        pickle.dump(
            data,
            f
        )


def update_claim_status(
    claim_number: int,
    new_status: str,
    reason: Optional[str] = None
):

    data = load_claim_status()

    claim_found = False

    for item in data:

        if int(
            item.get(
                "claim_number",
                -1
            )
        ) == int(claim_number):

            item["claim_status"] = new_status

            if reason is not None:
                item["reason"] = reason

            claim_found = True

            break

    if not claim_found:

        new_record = {
            "claim_number": int(claim_number),
            "claim_status": new_status
        }

        if reason is not None:
            new_record["reason"] = reason

        data.append(new_record)

    save_claim_status_data(data)



# CLAIM METADATA


def load_claim_metadata():

    if not os.path.exists(
        CLAIMS_METADATA_FILE
    ):
        return []

    try:

        with open(
            CLAIMS_METADATA_FILE,
            "rb"
        ) as f:

            metadata = pickle.load(f)

            if isinstance(metadata, list):
                return metadata

            return []

    except Exception:

        return []


def get_claim_documents(
    claim_number: int
):

    metadata = load_claim_metadata()

    claim_documents = []

    claimant_name = ""

    claim_amount = 0.0

    for item in metadata:

        if int(
            item.get(
                "claim_number",
                -1
            )
        ) == int(claim_number):

            claim_documents.append(item)

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

    return (
        claimant_name,
        claim_amount,
        claim_documents
    )



# POLICY RAG


def load_policy_context(
    query: str,
    top_k: int = 6
):

    if not os.path.exists(
        POLICY_FAISS_FILE
    ):

        return "Policy knowledge base is unavailable."

    if not os.path.exists(
        POLICY_METADATA_FILE
    ):

        return "Policy metadata is unavailable."

    try:

        policy_index = faiss.read_index(
            POLICY_FAISS_FILE
        )

        with open(
            POLICY_METADATA_FILE,
            "rb"
        ) as f:

            policy_metadata = pickle.load(f)

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

        scores, indices = policy_index.search(
            query_embedding,
            min(
                top_k,
                policy_index.ntotal
            )
        )

        policy_chunks = []

        for score, index_position in zip(
            scores[0],
            indices[0]
        ):

            if index_position < 0:
                continue

            if index_position >= len(
                policy_metadata
            ):
                continue

            item = policy_metadata[
                index_position
            ]

            text = item.get(
                "text",
                ""
            )

            if text:

                policy_chunks.append(
                    f"[Policy similarity: {score:.4f}]\n{text}"
                )

        if not policy_chunks:

            return "No relevant policy information was retrieved."

        return "\n\n".join(
            policy_chunks
        )

    except Exception as e:

        return (
            "Policy retrieval failed. "
            f"Error: {str(e)}"
        )



# DOCUMENT TEXT


def build_claim_document_text(
    documents
):

    sections = []

    for document in documents:

        document_type = document.get(
            "document_type",
            "unknown"
        )

        document_file = document.get(
            "document_file",
            "unknown"
        )

        text = document.get(
            "text",
            ""
        )

        sections.append(
            f"""
DOCUMENT TYPE: {document_type}
DOCUMENT FILE: {document_file}

{text}
"""
        )

    return "\n".join(
        sections
    )



# DOCUMENT VERIFICATION AGENT


def document_verification_agent(
    state: ClaimState
):

    claim_number = state[
        "claim_number"
    ]

    documents = state[
        "claim_documents"
    ]

    document_text = build_claim_document_text(
        documents
    )

    prompt = f"""
You are the Document Verification Agent for an insurance company.

Claim Number:
{claim_number}

Claimant Name:
{state["claimant_name"]}

Claim Amount:
Rs. {state["claim_amount"]:,.2f}

Claim documents:

{document_text}

Insurance policy rules:

- Government-issued identification is required.
- Policy number/certificate and completed claim form
  are required where applicable.
- Bills must show service provider, date,
  description, amount and invoice/bill number.
- Supporting documents depend on the claim type.
- Missing, unclear or inconsistent documents require
  human review.
- Do not assume that a document exists if it is not present.
- Do not hallicunate

Determine:

1. Whether identification is present.
2. Whether the identification appears valid.
3. Whether the claimant name reasonably matches
   the available identity information.
4. Whether a bill is present.
5. Whether the bill contains useful information.
6. Whether important claim information is missing.
7. Whether there are document inconsistencies.

Return ONLY valid JSON in this format:

{{
    "documents_complete": true,
    "identification_valid": true,
    "claimant_match": true,
    "bill_valid": true,
    "missing_documents": [],
    "document_issues": [],
    "reason": "short explanation"
}}
"""

    response = llm.invoke(prompt)

    result = parse_json_response(
        response.content
    )

    return {
        "document_result": result
    }



# ELIGIBILITY CHECK AGENT


def eligibility_check_agent(
    state: ClaimState
):

    claim_documents = state[
        "claim_documents"
    ]

    document_text = build_claim_document_text(
        claim_documents
    )

    policy_context = load_policy_context(
        document_text
    )

    prompt = f"""
You are the Eligibility Check Agent for an insurance company.

Claim Number:
{state["claim_number"]}

Claimant:
{state["claimant_name"]}

Claim Amount:
Rs. {state["claim_amount"]:,.2f}

Claim documents:

{document_text}

Relevant retrieved insurance policy:

{policy_context}

Eligibility rules:

1. Policy must be active on the incident date.
2. Claimant must be policyholder, insured person,
   nominee or authorized claimant.
3. Incident must be covered.
4. Claim must be reported within the applicable
   notification period unless valid delay is accepted.
5. Loss must have sufficient supporting evidence.
6. Loss must not be excluded.
7. Claim must be within applicable coverage limits.

Important:

Do not invent a policy number, policy date,
incident date or coverage information.

If required information cannot be established,
mark eligibility as "uncertain".

Return ONLY valid JSON:

{{
    "eligibility": "eligible",
    "policy_active": true,
    "claimant_covered": true,
    "incident_covered": true,
    "reported_on_time": true,
    "within_policy_limit": true,
    "excluded": false,
    "eligibility_issues": [],
    "reason": "short explanation"
}}

The eligibility field must be exactly one of:

"eligible"
"ineligible"
"uncertain"
"""

    response = llm.invoke(prompt)

    result = parse_json_response(
        response.content
    )

    return {
        "eligibility_result": result,
        "policy_context": policy_context
    }



# FRAUD DETECTION AGENT


def fraud_detection_agent(
    state: ClaimState
):

    documents = state[
        "claim_documents"
    ]

    document_text = build_claim_document_text(
        documents
    )

    prompt = f"""
You are the Fraud Detection Agent for an insurance company.

Claim Number:
{state["claim_number"]}

Claimant:
{state["claimant_name"]}

Claim Amount:
Rs. {state["claim_amount"]:,.2f}

Documents:

{document_text}

Look for potential fraud or irregularity indicators:

- Altered or forged documents
- Duplicate bills or claims
- Inconsistent dates
- Inconsistent names
- Inconsistent amounts
- Inconsistent incident details
- Unverifiable service providers
- Inflated or unreasonable claim amounts
- Loss before policy activation
- Repeated claim for the same loss
- False or misleading information
- Evidence that the loss may not have occurred
- Major discrepancies between documents

IMPORTANT:

A fraud indicator does NOT establish fraud.

Do not conclude that the claimant committed fraud.

If suspicious indicators exist, recommend human investigation.

Return ONLY valid JSON:

{{
    "fraud_risk": "low",
    "fraud_indicators": [],
    "document_manipulation_suspected": false,
    "duplicate_claim_suspected": false,
    "amount_concern": false,
    "reason": "short explanation"
}}

fraud_risk must be exactly:

"low"
"medium"
"high"
"""

    response = llm.invoke(prompt)

    result = parse_json_response(
        response.content
    )

    return {
        "fraud_result": result
    }



# JSON PARSER


def parse_json_response(
    content: str
):

    import json

    try:

        return json.loads(
            content
        )

    except Exception:

        # Try extracting JSON from markdown

        start = content.find("{")

        end = content.rfind("}")

        if start != -1 and end != -1:

            try:

                return json.loads(
                    content[start:end + 1]
                )

            except Exception:
                pass

        return {
            "error": "LLM did not return valid JSON",
            "raw_response": content
        }



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

    prompt = f"""
You are the Claim Summary Agent.

Create a concise insurance claim assessment summary.

Claim Number:
{state["claim_number"]}

Claimant:
{state["claimant_name"]}

Claim Amount:
Rs. {state["claim_amount"]:,.2f}

DOCUMENT VERIFICATION:

{document_result}

ELIGIBILITY:

{eligibility_result}

FRAUD ASSESSMENT:

{fraud_result}

Generate:

- Claim overview
- Document verification result
- Eligibility result
- Fraud assessment
- Important issues
- Recommended routing

Do not invent facts.

Return ONLY valid JSON:

{{
    "claim_overview": "",
    "document_assessment": "",
    "eligibility_assessment": "",
    "fraud_assessment": "",
    "important_issues": [],
    "recommended_routing": ""
}}
"""

    response = llm.invoke(
        prompt
    )

    result = parse_json_response(
        response.content
    )

    return {
        "summary_result": result
    }



# HUMAN APPROVAL AGENT


def human_approval_agent(
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
        state["claim_amount"]
    )

    reasons = []

    # High-value claims

    if claim_amount >= 10000:

        reasons.append(
            "Claim amount is Rs. 10,000 or more."
        )

    # Missing / unclear documents

    if not document_result.get(
        "documents_complete",
        False
    ):

        reasons.append(
            "Required documents are missing or incomplete."
        )

    # Document issues

    if document_result.get(
        "document_issues"
    ):

        reasons.append(
            "Document inconsistencies were identified."
        )

    # Eligibility uncertainty

    if eligibility_result.get(
        "eligibility"
    ) == "uncertain":

        reasons.append(
            "Claim eligibility could not be established."
        )

    # Fraud indicators

    fraud_risk = fraud_result.get(
        "fraud_risk",
        "low"
    )

    if fraud_risk in [
        "medium",
        "high"
    ]:

        reasons.append(
            "Potential fraud or irregularity indicators require investigation."
        )

    return {
        "human_result": {
            "required": True,
            "reasons": reasons,
            "message":
                "Human claims officer approval is required."
        },
        "processing_status":
            "Human Approval Required"
    }



# AUTOMATIC ROUTING


def route_after_parallel_checks(
    state: ClaimState
):

    claim_amount = float(
        state["claim_amount"]
    )

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

    
    # Missing / invalid documents
    

    if not document_result.get(
        "documents_complete",
        False
    ):

        return "reject"

    if not document_result.get(
        "identification_valid",
        False
    ):

        return "reject"

    if not document_result.get(
        "claimant_match",
        False
    ):

        return "human"

    if not document_result.get(
        "bill_valid",
        False
    ):

        return "reject"

    
    # Eligibility
    

    eligibility = eligibility_result.get(
        "eligibility"
    )

    if eligibility == "ineligible":

        return "reject"

    if eligibility == "uncertain":

        return "human"

    
    # Fraud
    

    fraud_risk = fraud_result.get(
        "fraud_risk",
        "low"
    )

    if fraud_risk in [
        "medium",
        "high"
    ]:

        return "human"

    
    # High value claim
    

    if claim_amount >= 10000:

        return "human"

    
    # Automatic approval
    

    return "approve"



# AUTO APPROVAL NODE


def automatic_approval_node(
    state: ClaimState
):

    reason = (
        "Claim is below Rs. 10,000 and the "
        "documents, eligibility and fraud checks "
        "satisfied the automatic approval rules."
    )

    return {
        "final_decision": "Approved",
        "final_reason": reason,
        "processing_status": "Approved"
    }



# REJECTION NODE


def rejection_node(
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

    reasons = []

    if not document_result.get(
        "documents_complete",
        True
    ):

        missing = document_result.get(
            "missing_documents",
            []
        )

        if missing:

            reasons.append(
                "Missing documents: "
                + ", ".join(missing)
            )

        else:

            reasons.append(
                "Required documents are incomplete."
            )

    if not document_result.get(
        "identification_valid",
        True
    ):

        reasons.append(
            "Identification could not be verified."
        )

    if not document_result.get(
        "bill_valid",
        True
    ):

        reasons.append(
            "Bill could not be verified."
        )

    if eligibility_result.get(
        "eligibility"
    ) == "ineligible":

        eligibility_issues = (
            eligibility_result.get(
                "eligibility_issues",
                []
            )
        )

        if eligibility_issues:

            reasons.extend(
                eligibility_issues
            )

        else:

            reasons.append(
                eligibility_result.get(
                    "reason",
                    "Claim does not satisfy eligibility requirements."
                )
            )

    if not reasons:

        reasons.append(
            "Claim does not satisfy the automatic processing requirements."
        )

    return {
        "final_decision": "Rejected",
        "final_reason": " ".join(reasons),
        "processing_status": "Rejected"
    }



# SAVE COMPLETE CLAIM RESULT


def save_claim_result(
    state: ClaimState
):

    if os.path.exists(
        CLAIM_RESULTS_FILE
    ):

        try:

            with open(
                CLAIM_RESULTS_FILE,
                "rb"
            ) as f:

                results = pickle.load(f)

        except Exception:

            results = []

    else:

        results = []

    result_record = {

        "claim_number":
            int(
                state["claim_number"]
            ),

        "claimant_name":
            state["claimant_name"],

        "claim_amount":
            float(
                state["claim_amount"]
            ),

        "document_result":
            state.get(
                "document_result",
                {}
            ),

        "eligibility_result":
            state.get(
                "eligibility_result",
                {}
            ),

        "fraud_result":
            state.get(
                "fraud_result",
                {}
            ),

        "summary_result":
            state.get(
                "summary_result",
                {}
            ),

        "human_result":
            state.get(
                "human_result",
                {}
            ),

        "final_decision":
            state.get(
                "final_decision",
                ""
            ),

        "final_reason":
            state.get(
                "final_reason",
                ""
            ),

        "processing_status":
            state.get(
                "processing_status",
                ""
            )
    }

    updated = False

    for i, item in enumerate(
        results
    ):

        if int(
            item.get(
                "claim_number",
                -1
            )
        ) == int(
            state["claim_number"]
        ):

            results[i] = result_record

            updated = True

            break

    if not updated:

        results.append(
            result_record
        )

    with open(
        CLAIM_RESULTS_FILE,
        "wb"
    ) as f:

        pickle.dump(
            results,
            f
        )



# FINALIZE APPROVAL / REJECTION


def finalize_human_decision(
    claim_number: int,
    decision: str,
    reason: str
):

    decision = decision.strip().lower()

    if decision == "approve":

        final_decision = "Approved"

        status = "Approved"

    elif decision == "reject":

        final_decision = "Rejected"

        status = "Rejected"

    else:

        raise ValueError(
            "Decision must be 'approve' or 'reject'."
        )

    claimant_name, claim_amount, documents = (
        get_claim_documents(
            claim_number
        )
    )

    state = ClaimState(

        claim_number=claim_number,

        claimant_name=claimant_name,

        claim_amount=claim_amount,

        claim_documents=documents,

        final_decision=final_decision,

        final_reason=reason,

        processing_status=status
    )

    # Save final status

    update_claim_status(
        claim_number,
        status,
        reason
    )

    # Update claim result

    save_claim_result(
        state
    )

    # Update claim metadata as well

    update_claim_metadata_status(
        claim_number,
        status
    )

    return state



# UPDATE CLAIM METADATA


def update_claim_metadata_status(
    claim_number: int,
    status: str
):

    metadata = load_claim_metadata()

    changed = False

    for item in metadata:

        if int(
            item.get(
                "claim_number",
                -1
            )
        ) == int(claim_number):

            item["claim_status"] = status

            changed = True

    if changed:

        with open(
            CLAIMS_METADATA_FILE,
            "wb"
        ) as f:

            pickle.dump(
                metadata,
                f
            )



# GRAPH NODES


def prepare_claim_node(
    state: ClaimState
):

    return {}



# BUILD LANGGRAPH


def build_claim_graph():

    graph = StateGraph(
        ClaimState
    )

    
    # Initial node
    

    graph.add_node(
        "prepare_claim",
        prepare_claim_node
    )

    
    # Required five agents/nodes
    

    graph.add_node(
        "document_verification",
        document_verification_agent
    )

    graph.add_node(
        "eligibility_check",
        eligibility_check_agent
    )

    graph.add_node(
        "fraud_detection",
        fraud_detection_agent
    )

    graph.add_node(
        "claim_summary",
        claim_summary_agent
    )

    graph.add_node(
        "human_approval",
        human_approval_agent
    )

    
    # Final processing nodes
    

    graph.add_node(
        "automatic_approval",
        automatic_approval_node
    )

    graph.add_node(
        "rejection",
        rejection_node
    )

    
    # START
    

    graph.add_edge(
        START,
        "prepare_claim"
    )

    
    # PARALLEL EXECUTION
    #
    # All three checks start independently after
    # prepare_claim.
    

    graph.add_edge(
        "prepare_claim",
        "document_verification"
    )

    graph.add_edge(
        "prepare_claim",
        "eligibility_check"
    )

    graph.add_edge(
        "prepare_claim",
        "fraud_detection"
    )

    
    # All three converge on Claim Summary
    

    graph.add_edge(
        "document_verification",
        "claim_summary"
    )

    graph.add_edge(
        "eligibility_check",
        "claim_summary"
    )

    graph.add_edge(
        "fraud_detection",
        "claim_summary"
    )

    
    # Conditional routing
    

    graph.add_conditional_edges(

        "claim_summary",

        route_after_parallel_checks,

        {
            "approve":
                "automatic_approval",

            "reject":
                "rejection",

            "human":
                "human_approval"
        }
    )

    
    # End states
    

    graph.add_edge(
        "automatic_approval",
        END
    )

    graph.add_edge(
        "rejection",
        END
    )

    graph.add_edge(
        "human_approval",
        END
    )

    return graph.compile()



# RUN CLAIM


def process_claim(
    claim_number: int
):

    claimant_name, claim_amount, documents = (
        get_claim_documents(
            claim_number
        )
    )

    if not documents:

        raise ValueError(
            f"Claim {claim_number} was not found "
            "in claim_metadata.pkl."
        )

    initial_state = ClaimState(

        claim_number=int(
            claim_number
        ),

        claimant_name=claimant_name,

        claim_amount=float(
            claim_amount
        ),

        claim_documents=documents,

        processing_status="Processing"
    )

    
    # Mark claim as processing
    

    update_claim_status(
        claim_number,
        "Processing"
    )

    
    # Build and invoke graph
    

    graph = build_claim_graph()

    result = graph.invoke(
        initial_state
    )

    
    # Save result
    

    save_claim_result(
        result
    )

    
    # Automatic approval / rejection
    #
    # Human review remains pending until a claims officer
    # makes a decision from Streamlit.
    

    if result.get(
        "processing_status"
    ) == "Approved":

        update_claim_status(
            claim_number,
            "Approved",
            result.get(
                "final_reason",
                ""
            )
        )

        update_claim_metadata_status(
            claim_number,
            "Approved"
        )

    elif result.get(
        "processing_status"
    ) == "Rejected":

        update_claim_status(
            claim_number,
            "Rejected",
            result.get(
                "final_reason",
                ""
            )
        )

        update_claim_metadata_status(
            claim_number,
            "Rejected"
        )

    elif result.get(
        "processing_status"
    ) == "Human Approval Required":

        update_claim_status(
            claim_number,
            "Human Approval Required",
            "; ".join(
                result.get(
                    "human_result",
                    {}
                ).get(
                    "reasons",
                    []
                )
            )
        )

        update_claim_metadata_status(
            claim_number,
            "Human Approval Required"
        )

    return result
