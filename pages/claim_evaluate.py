#Claim Processor & Langgraph

import streamlit as st
import os
import pickle
import json
import re
from typing import TypedDict, Optional, Any
import numpy as np
import faiss

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

#Referencing the Policy files
POLICY_FAISS_FILE = "insurance_policy.faiss"
POLICY_METADATA_FILE = "insurance_policy.pkl"

CLAIMS_FAISS_FILE = "claims.faiss"
CLAIMS_METADATA_FILE = "claim_metadata.pkl"

CLAIM_STATUS_FILE = "claim_status.pkl"

# New file created by the processing workflow
CLAIM_RESULTS_FILE = "claim_results.pkl"

#verifying the files
required_files = [
    POLICY_FAISS_FILE,
    POLICY_METADATA_FILE,
    CLAIMS_FAISS_FILE,
    CLAIMS_METADATA_FILE,
    CLAIM_STATUS_FILE
]

for file in required_files:
    print(
        f"{file}:",
        "FOUND" if os.path.exists(file) else "MISSING"
    )


#initialize LLM
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

# intialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


#creating generic Generic pickle helper functions - load pickle
def load_pickle(filename, default=None):

    if not os.path.exists(filename):
        return default

    try:
        with open(filename, "rb") as f:
            return pickle.load(f)

    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return default

#creating generic Generic pickle helper functions - save pickle
def save_pickle(filename, data):

    with open(filename, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved: {filename}")


#Function to update claim status
def update_claim_status(
    claim_number,
    status
):

    data = load_pickle(
        CLAIM_STATUS_FILE,
        []
    )

    claim_number = int(claim_number)

    found = False

    for item in data:

        if int(
            item.get("claim_number", -1)
        ) == claim_number:

            item["claim_status"] = status
            found = True
            break

    if not found:

        data.append(
            {
                "claim_number": claim_number,
                "claim_status": status
            }
        )

    save_pickle(
        CLAIM_STATUS_FILE,
        data
    )

    return data

#load claim status
claim_status_data = load_pickle(
    CLAIM_STATUS_FILE,
    []
)

#Load the claims metadata
claim_metadata = load_pickle(
    CLAIMS_METADATA_FILE,
    []
)

#Function to load a specific claim

def load_claim(
    claim_number
):

    claim_number = int(
        claim_number
    )

    metadata = load_pickle(
        CLAIMS_METADATA_FILE,
        []
    )

    records = [
        item
        for item in metadata
        if int(
            item.get(
                "claim_number",
                -1
            )
        ) == claim_number
    ]

    if not records:

        raise ValueError(
            f"Claim {claim_number} "
            f"was not found in "
            f"{CLAIMS_METADATA_FILE}"
        )

    claimant_name = records[0].get(
        "claimant_name",
        ""
    )

    claim_amount = float(
        records[0].get(
            "claim_amount",
            0
        )
    )

    return {
        "claim_number": claim_number,
        "claimant_name": claimant_name,
        "claim_amount": claim_amount,
        "documents": records
    }

#Group identification and bill documents
def group_documents(
    documents
):

    grouped = {}

    for document in documents:

        document_type = document.get(
            "document_type",
            "unknown"
        )

        if document_type not in grouped:

            grouped[
                document_type
            ] = []

        grouped[
            document_type
        ].append(document)

    return grouped


#Retrieve policy from FAISS
def retrieve_policy_context(
    claim_documents,
    top_k=8
):

    if not os.path.exists(
        POLICY_FAISS_FILE
    ):

        raise FileNotFoundError(
            POLICY_FAISS_FILE
        )

    if not os.path.exists(
        POLICY_METADATA_FILE
    ):

        raise FileNotFoundError(
            POLICY_METADATA_FILE
        )

    policy_index = faiss.read_index(
        POLICY_FAISS_FILE
    )

    policy_metadata = load_pickle(
        POLICY_METADATA_FILE,
        []
    )

    if not policy_metadata:

        return ""

    # Combine all claim document text.
    query_text = "\n".join(
        document.get(
            "text",
            ""
        )
        for document in claim_documents
    )

    if not query_text.strip():

        return ""

    # Generate query embedding.
    query_embedding = (
        embedding_model.embed_query(
            query_text
        )
    )

    query_embedding = np.asarray(
        [query_embedding],
        dtype="float32"
    )

    # Same normalization used when
    # the FAISS index was created.
    faiss.normalize_L2(
        query_embedding
    )

    k = min(
        top_k,
        policy_index.ntotal
    )

    if k == 0:

        return ""

    distances, indices = (
        policy_index.search(
            query_embedding,
            k
        )
    )

    context = []

    for index in indices[0]:

        if index < 0:
            continue

        if index >= len(policy_metadata):
            continue

        metadata = policy_metadata[index]

        text = metadata.get(
            "text",
            ""
        )

        if text.strip():

            context.append(
                text
            )

    return "\n\n".join(
        context
    )

#Return structured JSON from LLM
def invoke_json(
    prompt
):

    response = llm.invoke(
        prompt
    )

    content = response.content

    if not isinstance(
        content,
        str
    ):

        content = str(content)

    content = content.strip()

    # Remove ```json
    content = re.sub(
        r"```json\s*",
        "",
        content,
        flags=re.IGNORECASE
    )

    # Remove ```
    content = re.sub(
        r"```\s*$",
        "",
        content
    )

    content = content.strip()

    # First try direct JSON.
    try:

        return json.loads(
            content
        )

    except Exception:
        pass

    # If the model included additional text,
    # extract the JSON object.
    match = re.search(
        r"\{.*\}",
        content,
        re.DOTALL
    )

    if match:

        try:

            return json.loads(
                match.group(0)
            )

        except Exception:
            pass

    raise ValueError(
        "LLM did not return valid JSON:\n"
        + content
    )


#Defining LangGraph state
class ClaimState(TypedDict, total=False):
    claim_number: int
    claimant_name: str
    claim_amount: float
    claim_documents: list
    policy_context: str
    document_result: dict
    eligibility_result: dict
    fraud_result: dict
    claim_summary: dict
    final_decision: str
    final_reason: str
    human_required: bool
    human_decision: Optional[str]
    human_decision_reason: Optional[str]

#Agent to verify document: Node 1
def document_verification_agent(
    state: ClaimState
):
    documents = state.get(
        "claim_documents",
        []
    )

    grouped = group_documents(
        documents
    )

    identification_documents = (
        grouped.get(
            "identification",
            []
        )
    )

    bill_documents = (
        grouped.get(
            "bill",
            []
        )
    )

    all_text = "\n".join(
        document.get(
            "text",
            ""
        )
        for document in documents
    )

    prompt = f"""
You are the Document Verification Agent
for an insurance claim system.

Insurance document requirements:

1. At least one valid government-issued ID.
2. Policy number/certificate.
3. Completed claim form.
4. Depending on claim type, supporting bills/evidence.
5. Bills should contain:
   - service provider
   - date
   - description
   - amount
   - invoice/bill number

CLAIM DOCUMENTS:

{all_text}

Determine whether the supplied documents appear
complete and internally consistent.

Do not invent information.

Missing or unclear information is a document issue,
not proof of fraud.

If the bill has claim expiry date before current date then reject the bill.
If the bill has breakup and amount is less than Rs 10000 and indentification proof is present the approve the claim.
If not bill breakup is present then reject the claim.
Since this is testing, so If the ID has an identification number then consider to approve the claim.

Any claim having bill which is below Rs10000 and has invoice number, invoice date, claim name same as ID name, service, description, and service provider specified has to be auto approved.
Refer only the text content in the document.
Return ONLY JSON:

{{
    "documents_complete": true,
    "identification_present": true,
    "bill_present": true,
    "policy_document_present": true,
    "claim_form_present": true,
    "bill_details_complete": true,
    "consistent": true,
    "document_fraud_indicator": false,
    "missing_documents": [],
    "discrepancies": [],
    "reason": ""
}}

Use false where the information is clearly missing.
"""

    result = invoke_json(
        prompt
    )

    # Deterministic checks based on actual metadata.
    result[
        "identification_present"
    ] = bool(
        identification_documents
    )

    result[
        "bill_present"
    ] = bool(
        bill_documents
    )

    if "missing_documents" not in result:

        result[
            "missing_documents"
        ] = []

    if "discrepancies" not in result:

        result[
            "discrepancies"
        ] = []

    if not identification_documents:

        result[
            "missing_documents"
        ].append(
            "Government-issued identification"
        )

    if not bill_documents:

        result[
            "missing_documents"
        ].append(
            "Bill / supporting document"
        )

    result[
        "documents_complete"
    ] = (
        len(
            result["missing_documents"]
        ) == 0
        and result.get(
            "consistent",
            False
        )
        and result.get(
            "bill_details_complete",
            False
        )
    )

    return {
        "document_result": result
    }

#Agent to check Policy Eligibility: Node 2 
def eligibility_check_agent(
    state: ClaimState
):

    policy_context = state.get(
        "policy_context",
        ""
    )

    documents = state.get(
        "claim_documents",
        []
    )

    claim_amount = state.get(
        "claim_amount",
        0
    )

    claim_text = "\n".join(
        document.get(
            "text",
            ""
        )
        for document in documents
    )

    prompt = f"""
You are the Eligibility Check Agent
for an insurance claim.

Use the supplied insurance policy and
claim documents.

INSURANCE POLICY:

{policy_context}

CLAIM AMOUNT:

Rs. {claim_amount:,.2f}

CLAIM DOCUMENTS:

{claim_text}

Eligibility requirements:

1. Policy active on incident date.
2. Claimant is covered or authorized.
3. Incident is covered.
4. Claim reported within applicable period.
5. Sufficient evidence is supplied.
6. Loss is not excluded.
7. Claim is within coverage limits.

Do not assume missing information is true.
Restrict to policy guideline only.
If something cannot be established,
mark it as uncertain. 

If the bill has claim expiry date before current date then reject the bill.
If the bill has breakup and amount is less than Rs 10000 and indentification proof is present the approve the claim.
If not bill breakup is present then reject the claim.
Since this is testing, so If the ID has an identification number then consider to approve the claim.
Any claim having bill which is below Rs10000 and has invoice number, invoice date, claim name same as ID name, service, description, and service provider specified has to be auto approved.
Refer only the text content in the document.

Return ONLY JSON:

{{
    "eligible": true,
    "policy_active": true,
    "claimant_covered": true,
    "incident_covered": true,
    "reported_in_time": true,
    "excluded": false,
    "within_limit": true,
    "uncertain": false,
    "reasons": [],
    "policy_references": []
}}
"""

    result = invoke_json(
        prompt
    )

    if "reasons" not in result:

        result[
            "reasons"
        ] = []

    if "policy_references" not in result:

        result[
            "policy_references"
        ] = []

    if not policy_context.strip():

        result[
            "eligible"
        ] = False

        result[
            "uncertain"
        ] = True

        result[
            "reasons"
        ].append(
            "Policy information could not be retrieved."
        )

    return {
        "eligibility_result": result
    }



#Agent to Detect Frauds: Node 3
def fraud_detection_agent(
    state: ClaimState
):

    documents = state.get(
        "claim_documents",
        []
    )

    claim_amount = state.get(
        "claim_amount",
        0
    )

    claim_text = "\n".join(
        document.get(
            "text",
            ""
        )
        for document in documents
    )

    prompt = f"""
You are the Fraud Detection Agent
for an insurance claim system.

A fraud indicator does NOT establish fraud.

If the bill has claim expiry date before current date then reject the bill.
If the bill has breakup and amount is less than Rs 10000 and indentification proof is present the approve the claim.
If not bill breakup is present then reject the claim.
Since this is testing, so If the ID has an identification number then consider to approve the claim.
Any claim having bill which is below Rs10000 and has invoice number, invoice date, claim name same as ID name, service, description, and service provider specified has to be auto approved.

Refer only the text content in the document.

Identify potential irregularities requiring
human investigation.

Claim amount:

Rs. {claim_amount:,.2f}

Claim documents:

{claim_text}

Check for:

- altered documents
- forged documents
- duplicate bills
- inconsistent dates
- inconsistent names
- inconsistent amounts
- inconsistent incident details
- unverifiable providers
- inflated amounts
- loss before policy activation
- repeated claim
- misleading information
- evidence the loss may not have occurred
- major discrepancies

Return ONLY JSON:

{{
    "fraud_indicator": false,
    "risk_level": "LOW",
    "indicators": [],
    "inconsistencies": [],
    "reason": ""
}}
"""

    result = invoke_json(
        prompt
    )

    if "indicators" not in result:

        result[
            "indicators"
        ] = []

    if "inconsistencies" not in result:

        result[
            "inconsistencies"
        ] = []

    return {
        "fraud_result": result
    }

#Agent to generate claim summary: Node 4
def claim_summary_agent(
    state: ClaimState
):

    claim_number = state[
        "claim_number"
    ]

    claimant_name = state[
        "claimant_name"
    ]

    claim_amount = state[
        "claim_amount"
    ]

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
Create an insurance claim assessment summary.

Claim Number:
{claim_number}

Claimant:
{claimant_name}

Claim Amount:
Rs. {claim_amount:,.2f}

DOCUMENT VERIFICATION:

{json.dumps(
    document_result,
    indent=2
)}

ELIGIBILITY:

{json.dumps(
    eligibility_result,
    indent=2
)}

FRAUD ASSESSMENT:

{json.dumps(
    fraud_result,
    indent=2
)}

Return ONLY JSON:

{{
    "summary": "",
    "key_findings": [],
    "issues": []
}}
"""

    summary = invoke_json(
        prompt
    )

    return {
        "claim_summary": summary
    }

#Agent to decide whether the claim should be automatically approved, rejected or escalated for human review 
def automatic_decision_agent(
    state: ClaimState
):

    claim_amount = float(
        state.get(
            "claim_amount",
            0
        )
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

    # RULE 1
    # Rs. 10,000 or more -> human approval
    if claim_amount >= 10000:

        return {
            "final_decision":
                "Human Approval Required",

            "final_reason":
                "Claim amount is Rs. 10,000 or more.",

            "human_required":
                True
        }

    # RULE 2
    # Missing/invalid documents -> reject

    if not document_result.get(
        "documents_complete",
        False
    ):

        return {
            "final_decision":
                "Rejected",

            "final_reason":
                "Required documents are missing, "
                "invalid, incomplete or inconsistent.",

            "human_required":
                False
        }

    # RULE 3
    # Eligibility uncertain -> human

    if eligibility_result.get(
        "uncertain",
        False
    ):

        return {
            "final_decision":
                "Human Approval Required",

            "final_reason":
                "Claim eligibility could not be established "
                "with sufficient certainty.",

            "human_required":
                True
        }

    # RULE 4
    # Definitely ineligible -> reject

    if not eligibility_result.get(
        "eligible",
        False
    ):

        return {
            "final_decision":
                "Rejected",

            "final_reason":
                "Claim does not satisfy policy eligibility "
                "requirements.",

            "human_required":
                False
        }
    # RULE 5
    # Fraud indicator -> human investigation

    if fraud_result.get(
        "fraud_indicator",
        False
    ):

        return {
            "final_decision":
                "Human Approval Required",

            "final_reason":
                "Potential fraud or document irregularity "
                "requires human investigation.",

            "human_required":
                True
        }

    # RULE 6
    # Automatic approval

    return {
        "final_decision":
            "Approved",

        "final_reason":
            "Claim is below Rs. 10,000, eligible, "
            "documents are complete and consistent, "
            "and no significant fraud indicators "
            "were identified.",

        "human_required":
            False
    }

#Human agent Node 5
def human_approval_agent(
    state: ClaimState
):

    claim_number = state["claim_number"]

    claim_amount = state["claim_amount"]

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

    # Build reasons for human review

    reasons = []

    if claim_amount >= 10000:

        reasons.append(
            "Claim amount is Rs. 10,000 or more."
        )

    if not document_result.get(
        "documents_complete",
        False
    ):

        reasons.append(
            "Documents require human verification."
        )

    if eligibility_result.get(
        "uncertain",
        False
    ):

        reasons.append(
            "Eligibility is uncertain."
        )

    if fraud_result.get(
        "fraud_indicator",
        False
    ):

        reasons.append(
            "Potential fraud indicators require investigation."
        )

    # Display claim information

    print("\n" + "=" * 60)

    print("HUMAN APPROVAL REQUIRED")

    print("=" * 60)

    print(
        "Claim Number:",
        claim_number
    )

    print(
        "Claim Amount:",
        f"Rs. {claim_amount:,.2f}"
    )

    print("\nReasons for human review:")

    for reason in reasons:

        print(
            "-",
            reason
        )

    print("=" * 60)

    # Ask human for decision

    decision = input(
        "\nApprove claim? (approve / reject): "
    )

    decision = decision.strip().lower()

    # Validate input

    while decision not in {
        "approve",
        "reject"
    }:

        print(
            "\nInvalid input."
        )

        print(
            "Please enter: approve or reject"
        )

        decision = input(
            "\nApprove claim? (approve / reject): "
        )

        decision = decision.strip().lower()

    # Ask for reason

    human_reason = input(
        "\nEnter approval/rejection reason "
        "(optional): "
    ).strip()

    # APPROVE

    if decision == "approve":

        final_reason = (
            human_reason
            if human_reason
            else
            "Claim approved by authorized human reviewer."
        )

        return {

            "final_decision":
                "Approved",

            "final_reason":
                final_reason,

            "human_required":
                True,

            "human_decision":
                "approve",

            "human_decision_reason":
                final_reason
        }

    # REJECT

    else:

        final_reason = (
            human_reason
            if human_reason
            else
            "Claim rejected by authorized human reviewer."
        )

        return {

            "final_decision":
                "Rejected",

            "final_reason":
                final_reason,

            "human_required":
                True,

            "human_decision":
                "reject",

            "human_decision_reason":
                final_reason
        }

#Save result
def save_claim_result(
    result
):

    results = load_pickle(
        CLAIM_RESULTS_FILE,
        []
    )

    claim_number = int(
        result["claim_number"]
    )

    found = False

    for index, item in enumerate(
        results
    ):

        if int(
            item.get(
                "claim_number",
                -1
            )
        ) == claim_number:

            results[index] = result
            found = True
            break

    if not found:

        results.append(
            result
        )

    save_pickle(
        CLAIM_RESULTS_FILE,
        results
    )

#Persistence node
def persist_result_node(
    state: ClaimState
):

    claim_number = int(
        state["claim_number"]
    )

    decision = state.get(
        "final_decision",
        "Unknown"
    )

    if decision == "Approved":

        status = "Approved"

    elif decision == "Rejected":

        status = "Rejected"

    elif decision == "Human Approval Required":

        status = "Human Approval Required"

    else:

        status = decision

    result = {

        "claim_number":
            claim_number,

        "claimant_name":
            state.get(
                "claimant_name",
                ""
            ),

        "claim_amount":
            state.get(
                "claim_amount",
                0
            ),

        "claim_status":
            status,

        "final_decision":
            decision,

        "final_reason":
            state.get(
                "final_reason",
                ""
            ),

        "human_decision":
            state.get(
                "human_decision",
                None
            ),

        "human_decision_reason":
            state.get(
                "human_decision_reason",
                ""
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

        "claim_summary":
            state.get(
                "claim_summary",
                {}
            )
    }

    # Save detailed processing result

    save_claim_result(
        result
    )

    # Update claim_status.pkl

    update_claim_status(
        claim_number,
        status
    )

    return {}

#Node to decide which node executes after the summary
def route_after_summary(
    state: ClaimState
):

    claim_amount = float(
        state.get(
            "claim_amount",
            0
        )
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

    # Rs. 10,000 or more

    if claim_amount >= 10000:

        return "human_approval"

    # Uncertain eligibility

    if eligibility_result.get(
        "uncertain",
        False
    ):

        return "human_approval"

    # Fraud indicator

    if fraud_result.get(
        "fraud_indicator",
        False
    ):

        return "human_approval"

    # Everything else goes through automatic decision.

    return "automatic_decision"

#Build Langgraph
def build_claim_graph():

    graph = StateGraph(
        ClaimState
    )

    # ADD NODES

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
        "automatic_decision",
        automatic_decision_agent
    )

    graph.add_node(
        "human_approval",
        human_approval_agent
    )

    graph.add_node(
        "persist_result",
        persist_result_node
    )

    # START -> THREE PARALLEL BRANCHES

    graph.add_edge(
        START,
        "document_verification"
    )

    graph.add_edge(
        START,
        "eligibility_check"
    )

    graph.add_edge(
        START,
        "fraud_detection"
    )

    # THREE BRANCHES -> SUMMARY

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

    # SUMMARY -> CONDITIONAL ROUTING

    graph.add_conditional_edges(

        "claim_summary",

        route_after_summary,

        {
            "automatic_decision":
                "automatic_decision",

            "human_approval":
                "human_approval"
        }
    )

    # DECISION -> PERSIST

    graph.add_edge(
        "automatic_decision",
        "persist_result"
    )

    graph.add_edge(
        "human_approval",
        "persist_result"
    )

    # PERSIST -> END

    graph.add_edge(
        "persist_result",
        END
    )

    memory = MemorySaver()

    return graph.compile(
        checkpointer=memory
    )

claim_graph = build_claim_graph()

#Main function to process claim
def process_claim(
    claim_number
):

    claim_number = int(
        claim_number
    )

    # LOAD CLAIM

    claim = load_claim(
        claim_number
    )

    print(
        f"Processing Claim {claim_number}"
    )

    print(
        f"Claimant: {claim['claimant_name']}"
    )

    print(
        f"Amount: Rs. {claim['claim_amount']:,.2f}"
    )

    # RETRIEVE POLICY

    print(
        "\nRetrieving relevant policy..."
    )

    policy_context = retrieve_policy_context(
        claim["documents"]
    )

    # INITIAL STATE

    initial_state = {

        "claim_number":
            claim["claim_number"],

        "claimant_name":
            claim["claimant_name"],

        "claim_amount":
            claim["claim_amount"],

        "claim_documents":
            claim["documents"],

        "policy_context":
            policy_context
    }

    # THREAD ID

    config = {

        "configurable": {

            "thread_id":
                f"claim-{claim_number}"
        }
    }

    # EXECUTE GRAPH

    print(
        "\nExecuting LangGraph..."
    )
    
    result = claim_graph.invoke(
        initial_state,
        config=config
    )

    # CHECK FOR INTERRUPT

    state_snapshot = claim_graph.get_state(
        config
    )

    if state_snapshot.interrupts:

        interrupt_data = (
            state_snapshot.interrupts[0].value
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "HUMAN APPROVAL REQUIRED"
        )

        print(
            "=" * 60
        )

        print(
            "Claim Number:",
            interrupt_data.get(
                "claim_number"
            )
        )

        print(
            "Claim Amount:",
            f"Rs. {interrupt_data.get('claim_amount', 0):,.2f}"
        )

        print(
            "\nReasons:"
        )

        for reason in interrupt_data.get(
            "reasons",
            []
        ):

            print(
                "-",
                reason
            )

        print(
            "\nOptions:"
        )

        print(
            "1. approve"
        )

        print(
            "2. reject"
        )

        print(
            "=" * 60
        )

        return {
            "status":
                "Human Approval Required",

            "claim_number":
                claim_number,

            "interrupt":
                interrupt_data,

            "config":
                config,

            "state":
                result
        }

    # NORMAL COMPLETION

    print(
        "\n" + "=" * 60
    )

    print(
        "CLAIM PROCESSING COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "Claim Number:",
        result.get(
            "claim_number"
        )
    )

    print(
        "Status:",
        result.get(
            "final_decision"
        )
    )

    print(
        "Reason:",
        result.get(
            "final_reason"
        )
    )

    print(
        "=" * 60
    )

    return result