import streamlit as st
from typing import List, Dict, Optional

def display_chat_message(message: str, is_user: bool = True, sources: Optional[List[Dict]] = None):
    """Display a chat message with proper styling"""
    if is_user:
        with st.container():
            st.markdown("**You:**")
            st.info(message)
    else:
        with st.container():
            st.markdown("**Assistant:**")
            st.success(message)
        if sources:
            st.markdown("**Sources:**")
            for source in sources:
                with st.expander(f"📄 {source['source']}", expanded=True):
                    st.write(f"**Content:** {source['content']}")
