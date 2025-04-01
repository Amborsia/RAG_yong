def apply_custom_styles():
    """커스텀 CSS 스타일 적용"""
    import streamlit as st

    st.markdown(
        """
        <style>
        .stMain { position: relative; }
        .stChatMessage { background-color: transparent !important; }
        .st-emotion-cache-glsyku { align-items: center; }
        [data-testid="stChatMessage"]:has(> [data-testid="stChatMessageAvatarUser"]) {
            justify-content: flex-end !important;
            display: flex !important;
        }
        [data-testid="stChatMessage"]:has(> [data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            text-align: right !important;
            background-color: #3399FF !important;
            color: #FFFFFF !important;
            border-radius: 10px !important;
            padding: 10px !important;
            margin: 5px 0 !important;
            max-width: 80% !important;
            flex-grow: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
