import streamlit as st
from typing import List, Dict, Optional

def display_chat_message(message: str, is_user: bool = True, sources: Optional[List[Dict]] = None, retrieval_method: Optional[str] = None):
    """Display a chat message with proper styling and enhanced metadata"""
    if is_user:
        with st.container():
            st.markdown("**You:**")
            st.info(message)
    else:
        with st.container():
            st.markdown("**Assistant:**")
            st.success(message)
            
            # Display retrieval method if available
            if retrieval_method and retrieval_method != "unknown":
                method_colors = {
                    "mmr": "🔍",
                    "similarity": "📊", 
                    "hybrid": "🔗"
                }
                method_icon = method_colors.get(retrieval_method, "🔍")
                st.caption(f"{method_icon} Retrieved using {retrieval_method.upper()} search")
        
        if sources:
            st.markdown("**Sources:**")
            for i, source in enumerate(sources, 1):
                # Get content type and position info
                content_type = source.get('content_type', 'unknown')
                chunk_position = source.get('chunk_position', 'unknown')
                
                # Create expander title with enhanced info
                title = f"📄 {source['source']}"
                if content_type != 'unknown':
                    title += f" ({content_type})"
                if chunk_position != 'unknown':
                    title += f" - Chunk {chunk_position}"
                
                with st.expander(title, expanded=True):
                    st.write(f"**Content:** {source['content']}")
                    
                    # Display additional metadata if available
                    metadata_info = []
                    if content_type != 'unknown':
                        metadata_info.append(f"**Type:** {content_type}")
                    if chunk_position != 'unknown':
                        metadata_info.append(f"**Position:** {chunk_position}")
                    
                    if metadata_info:
                        st.markdown(" | ".join(metadata_info))
