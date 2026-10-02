from langchain.agents import create_agent
from llm_gpt import llm


VALID_INTENTS = {
    "UPDATE_INSURANCE_POLICY",
    "RAISE_CLAIM",
    "CLAIM_SUMMARY",
    "CLAIM_EVALUATE",
    "INVALID_INTENT",
}


def detect_intent(user_message: str) -> str:

    agent = create_agent(
        model=llm,
        system_prompt="""
        You are a routing assistant for an AI insurance claims application.

        Determine the user's intent.

        You MUST return exactly ONE of these values:

        UPDATE_INSURANCE_POLICY
        RAISE_CLAIM
        CLAIM_SUMMARY
        CLAIM_EVALUATE
        INVALID_INTENT

        Rules:

        - User wants to update, modify, change, or edit an insurance policy:
        UPDATE_INSURANCE_POLICY

        Examples:
        "I want to update my insurance policy"
        UPDATE_INSURANCE_POLICY

        "Change my policy details"
        UPDATE_INSURANCE_POLICY

        "I need to modify my insurance coverage"
        UPDATE_INSURANCE_POLICY

        "Update my policy"
        UPDATE_INSURANCE_POLICY


        - User wants to raise, submit, file, or create a NEW insurance claim:
        RAISE_CLAIM

        Examples:
        "I want to raise a claim"
        RAISE_CLAIM

        "I need to file an insurance claim"
        RAISE_CLAIM

        "Submit a new claim"
        RAISE_CLAIM

        "I want to report an accident and make a claim"
        RAISE_CLAIM


        - User wants to see, view, or get a summary of their insurance claims:
        CLAIM_SUMMARY

        Examples:
        "Show me my claims"
        CLAIM_SUMMARY

        "I want to see my claim summary"
        CLAIM_SUMMARY

        "Show my insurance claim history"
        CLAIM_SUMMARY

        "Give me a summary of my claims"
        CLAIM_SUMMARY


        - User wants to evaluate, assess, review, or analyze an insurance claim:
        CLAIM_EVALUATE

        Examples:
        "Evaluate my claim"
        CLAIM_EVALUATE

        "I want to assess this claim"
        CLAIM_EVALUATE

        "Review my insurance claim"
        CLAIM_EVALUATE

        "Analyze the claim"
        CLAIM_EVALUATE


        - If the request does not match any of the above:
        INVALID_INTENT

        Important distinctions:

        - Updating an existing policy -> UPDATE_INSURANCE_POLICY
        - Creating or submitting a new claim -> RAISE_CLAIM
        - Viewing claim information/history/summary -> CLAIM_SUMMARY
        - Evaluating or analyzing a claim -> CLAIM_EVALUATE

        Do not explain your answer.
        Return ONLY the intent value.
        """
    )

    # create_agent expects a state dictionary.
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_message,
                }
            ]
        }
    )

    # LangGraph returns the agent state.
    messages = response.get("messages", [])

    if not messages:
        return "INVALID_INTENT"

    # Last message is the agent's response.
    content = messages[-1].content

    # Normalize model output.
    intent = content.strip().upper()

    # Handle accidental formatting such as:
    # "RAISE_CLAIM."
    # "`RAISE_CLAIM`"
    intent = intent.replace("`", "")
    intent = intent.replace(".", "")
    intent = intent.strip()

    # Exact valid intent.
    if intent in VALID_INTENTS:
        return intent

    # Handle common natural-language outputs.
    if intent in {
        "UPDATE POLICY",
        "UPDATE INSURANCE POLICY",
        "MODIFY POLICY",
        "CHANGE POLICY",
        "EDIT POLICY",
    }:
        return "UPDATE_INSURANCE_POLICY"

    if intent in {
        "RAISE CLAIM",
        "FILE CLAIM",
        "SUBMIT CLAIM",
        "NEW CLAIM",
        "CREATE CLAIM",
        "INSURANCE CLAIM",
    }:
        return "RAISE_CLAIM"

    if intent in {
        "CLAIM SUMMARY",
        "CLAIMS SUMMARY",
        "SHOW CLAIMS",
        "VIEW CLAIMS",
        "CLAIM HISTORY",
        "CLAIMS HISTORY",
        "SHOW CLAIM SUMMARY",
    }:
        return "CLAIM_SUMMARY"

    if intent in {
        "CLAIM EVALUATE",
        "EVALUATE CLAIM",
        "EVALUATE CLAIMS",
        "ASSESS CLAIM",
        "ASSESS CLAIMS",
        "REVIEW CLAIM",
        "REVIEW CLAIMS",
        "ANALYZE CLAIM",
        "ANALYZE CLAIMS",
    }:
        return "CLAIM_EVALUATE"

    return "INVALID_INTENT"
