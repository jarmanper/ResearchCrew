# Deep Research Agent
# Streamlit app that runs AI research agents
# Uses Ollama locally, switches to Groq when deployed to the cloud

import streamlit as st
import os
from crew import ResearchCrew


# Generates Material Icon HTML for cleaner UI elements
def icon(name, size="18px"):
    return f'<span class="material-icons" style="font-size:{size};vertical-align:middle;margin-right:6px;">{name}</span>'


# Check if we're running on the cloud or locally
# If there's a Groq API key in secrets, we're on the cloud
if "GROQ_API_KEY" in st.secrets:
    env_mode = "Cloud"
    api_key = st.secrets["GROQ_API_KEY"]
    
    # Current Groq models (as of Jan 2026)
    # Note: Model availability changes - check console.groq.com for latest
    model_options = {
        "groq/llama-3.3-70b-versatile": "Llama 3.3 70B (Recommended)",
        "groq/llama-3.1-8b-instant": "Llama 3.1 8B (Fast)",
        "groq/mixtral-8x7b-32768": "Mixtral 8x7B"
    }
    
    # Set the env var so litellm can find it
    os.environ["GROQ_API_KEY"] = api_key
else:
    env_mode = "Local"
    api_key = None
    
    # Ollama models with litellm prefix
    model_options = {
        "ollama/deepseek-r1:8b": "DeepSeek R1 8B",
        "ollama/llama3.2": "Llama 3.2",
        "ollama/mistral": "Mistral"
    }


# Set up the page
st.set_page_config(
    page_title="Deep Research Agent", 
    page_icon=None,
    layout="centered"
)


# Custom CSS for styling
st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<style>
    .stAppDeployButton {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stMarkdown {padding: 20px; border-radius: 10px;}
    .icon-text {display: flex; align-items: center;}
</style>
""", unsafe_allow_html=True)


# Header
st.markdown(f'<h1>{icon("policy", "32px")} Deep Research Agent</h1>', unsafe_allow_html=True)
st.markdown(f"### Mode: **{env_mode}**")

if env_mode == "Local":
    st.info("Running on Local Compute. Using your local Ollama instance.")
else:
    st.info("Running on Cloud Compute. Using Groq's cloud acceleration.")


# Sidebar with settings
with st.sidebar:
    st.markdown(f'<h3>{icon("settings", "24px")} Configuration</h3>', unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Choose Model", 
        options=list(model_options.keys()), 
        format_func=lambda x: model_options[x],
        index=0
    )
    
    temperature = st.slider(
        "Creativity Level", 
        0.0, 1.0, 0.7, 
        help="Higher = More Creative, Lower = More Factual"
    )
    
    st.markdown("---")
    
    topic = st.text_input("Research Topic", "The Future of AI in 2026")
    
    st.markdown("---")
    
    run_btn = st.button("Launch Research Crew", type="primary", use_container_width=True)


# Run the crew when button is clicked
if run_btn:
    with st.status("**Agents are working...**", expanded=True) as status:
        try:
            st.markdown(f'{icon("search")} Researcher Agent is gathering data...', unsafe_allow_html=True)
            
            crew = ResearchCrew(
                topic=topic, 
                model_name=selected_model, 
                temperature=temperature,
                api_key=api_key
            )
            
            st.markdown(f'{icon("edit_note")} Writer Agent is drafting the content...', unsafe_allow_html=True)
            
            result = crew.run()
            
            status.update(label="Research Complete!", state="complete", expanded=False)
            
            st.markdown("---")
            st.markdown(f'<h3>{icon("article", "24px")} Report: {topic}</h3>', unsafe_allow_html=True)
            st.markdown(result)
            
            st.download_button(
                label="Download Report",
                data=str(result),
                file_name=f"{topic.replace(' ', '_')}_report.md",
                mime="text/markdown"
            )

        except Exception as e:
            status.update(label="Error Occurred", state="error")
            st.error(f"An error occurred: {e}")
