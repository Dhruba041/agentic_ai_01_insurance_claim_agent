# Screen to upload files and respond to insurance queries

import streamlit as st
from detect_intent import detect_intent
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import os


load_dotenv()

st.set_page_config(
    page_title="Insurance AI Assistant",
    page_icon="🛡️",
    layout="wide"
)


# Check Streamlit secrets before accessing the key
if "OPENAI_API_KEY" not in st.secrets:
    st.error(
        "OpenAI API key is not configured. "
        "Please add OPENAI_API_KEY to Streamlit secrets."
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)



# Custom CSS


st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #00FFFF;
    }

    /* Assistant message */
    div[data-testid="stChatMessage"]:has(
        div[data-testid="stChatMessageAvatarAssistant"]
    ) {
        background-color: #00FFFF;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
    }

    /* Assistant text */
    div[data-testid="stChatMessage"]:has(
        div[data-testid="stChatMessageAvatarAssistant"]
    ) div[data-testid="stMarkdownContainer"] {
        color: #6B3E26 !important;
    }

    /* User message */
    div[data-testid="stChatMessage"]:has(
        div[data-testid="stChatMessageAvatarUser"]
    ) {
        background-color: #FFF3B0;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
    }

    /* User text */
    div[data-testid="stChatMessage"]:has(
        div[data-testid="stChatMessageAvatarUser"]
    ) div[data-testid="stMarkdownContainer"] {
        color: #003366 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #7CFC00;
    }

    /* Spinner */
    .stSpinner > div {
        color: #003366 !important;
    }

    .stSpinner svg {
        stroke: #003366 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# Session State Initialization


if "messages" not in st.session_state:
    st.session_state.messages = []

if "greeting_shown" not in st.session_state:
    st.session_state.greeting_shown = False

if "session_active" not in st.session_state:
    st.session_state.session_active = True



# Initial Greeting


if not st.session_state.greeting_shown:

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 Welcome to the Insurance AI Assistant.\n\n"
                "I can help you with:\n\n"
                "• Updating your insurance policy\n"
                "• Raising a new insurance claim\n"
                "• Viewing your claim summary\n"
                "• Processing/evaluating an insurance claim\n\n"
                "Please enter what you would like to do.\n\n"
                "For example:\n"
                "• Update my insurance policy\n"
                "• Raise a new claim\n"
                "• Show my claim summary\n"
                "• Evaluate my claim"
            )
        }
    )

    st.session_state.greeting_shown = True



# Display Chat History


for message in st.session_state.messages:

    role = message["role"]
    content = message["content"]

    if role == "user":

        with st.chat_message("user"):

            st.markdown(
                f'<div style="color:#003366;">{content}</div>',
                unsafe_allow_html=True
            )

    elif role == "assistant":

        with st.chat_message("assistant"):

            st.markdown(
                f'<div style="color:#6B3E26;">{content}</div>',
                unsafe_allow_html=True
            )



# Chat Input


user_input = st.chat_input(
    "Update policy, raise a claim, view claim summary, or process a claim. "
    "Type 'clear' to clear chat or 'exit' to exit.",
    disabled=not st.session_state.session_active
)



# Process Chat Input


if user_input and st.session_state.session_active:

    command = user_input.strip().lower()


    
    # EXIT
    

    if command == "exit":

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "Goodbye! 👋\n\n"
                    "Thank you for using the Insurance AI Assistant.\n\n"
                    "This session is now inactive.\n\n"
                    "Please close the browser or refresh the page "
                    "to start a new session.\n\n"
                    "Have a great day!"
                )
            }
        )

        st.session_state.session_active = False

        st.rerun()


    
    # CLEAR CHAT
    

    elif command == "clear":

        st.session_state.messages = []

        st.session_state.greeting_shown = False

        st.rerun()


    
    # INTENT DETECTION
    

    else:

        # Store user's message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        # Detect intent
        intent_response = detect_intent(user_input)


        
        # UPDATE INSURANCE POLICY
        

        if intent_response == "UPDATE_INSURANCE_POLICY":

            st.session_state["navigation_message"] = (
                "Taking you to the Insurance Policy update page..."
            )

            st.switch_page(
                "pages/insurance_policy_app.py"
            )


        
        # RAISE NEW CLAIM
        

        elif intent_response == "RAISE_CLAIM":

            st.session_state["navigation_message"] = (
                "Taking you to the Claim submission page..."
            )

            st.switch_page(
                "pages/claim_app.py"
            )


        
        # CLAIM SUMMARY
        

        elif intent_response == "CLAIM_SUMMARY":

            st.session_state["navigation_message"] = (
                "Taking you to the Claim Summary page..."
            )

            st.switch_page(
                "pages/claim_summary_app.py"
            )


        
        # CLAIM EVALUATION / PROCESSING
        

        elif intent_response == "CLAIM_EVALUATE":

            st.session_state["navigation_message"] = (
                "Taking you to the Claim Processing page..."
            )

            st.switch_page(
                "pages/claim_process_app.py"
            )


        
        # INVALID INTENT
        

        else:

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "❌ Invalid input.\n\n"
                        "I can help you with the following:\n\n"
                        "• Update your insurance policy\n"
                        "• Raise a new insurance claim\n"
                        "• View your claim summary\n"
                        "• Process or evaluate an insurance claim\n\n"
                        "Please enter one of these requests."
                    )
                }
            )

            st.rerun()
