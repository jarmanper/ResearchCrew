"""
Deep Research Agent - A Streamlit-based research assistant powered by AI agents.

This app brings together a team of AI agents (researcher and writer) to tackle 
any research topic you throw at them. Just type in what you want to learn about,
and let the agents do the heavy lifting while you grab a coffee.

Now runs as a hybrid app - uses local Ollama on your machine, but automatically
switches to Groq cloud when deployed to Streamlit Cloud.
"""

import streamlit as st
import os
from crew import ResearchCrew


# Helper function for generating Material Icons in our UI
def icon(name, size="18px"):
    """
    Generates an inline Material Icon HTML snippet.
    
    We use Material Icons instead of emojis for a cleaner, more professional look.
    The icons are loaded from Google Fonts and render beautifully at any size.
    """
    return f'<span class="material-icons" style="font-size:{size};vertical-align:middle;margin-right:6px;">{name}</span>'


# Smart environment detection - figures out if we're running locally or in the cloud
# If there's a Groq API key in Streamlit secrets, we're on the cloud
if "GROQ_API_KEY" in st.secrets:
    # Cloud mode - use Groq's blazing fast inference
    env_mode = "Cloud"
    api_key = st.secrets["GROQ_API_KEY"]
    base_url = "https://api.groq.com/openai/v1"
    
    # Groq has its own model names - DeepSeek R1 is available as a distilled version
    model_options = {
        "deepseek-r1-distill-llama-70b": "Thinking (DeepSeek R1 - Cloud)",
        "llama-3.3-70b-versatile": "Fast (Llama 3.3 - Cloud)",
        "mixtral-8x7b-32768": "Pro (Mixtral - Cloud)"
    }
else:
    # Local mode - use Ollama running on your machine
    env_mode = "Local"
    api_key = "NA"
    base_url = "http://localhost:11434/v1"
    
    # These are the Ollama model names - make sure you've pulled them first
    model_options = {
        "deepseek-r1:8b": "Thinking (DeepSeek R1 - Local)",
        "llama3.2": "Fast (Llama 3.2 - Local)",
        "mistral": "Pro (Mistral - Local)"
    }

# Set a dummy API key for libraries that expect one even when using local models
os.environ["OPENAI_API_KEY"] = api_key


# Page config needs to be the first Streamlit command
st.set_page_config(
    page_title="Deep Research Agent", 
    page_icon=None,
    layout="centered"
)


# Custom styling to polish up the default Streamlit look
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


# Main header - the first thing users see
st.markdown(f'<h1>{icon("policy", "32px")} Deep Research Agent</h1>', unsafe_allow_html=True)
st.markdown(f"### Mode: **{env_mode}**")

# Let users know which backend they're using
if env_mode == "Local":
    st.info("Running on Local Compute. Using your local Ollama instance.")
else:
    st.info("Running on Cloud Compute. Using Groq's cloud acceleration.")


# Sidebar holds all the configuration options
with st.sidebar:
    st.markdown(f'<h3>{icon("settings", "24px")} Configuration</h3>', unsafe_allow_html=True)

    # Model dropdown - options change based on whether we're local or cloud
    selected_model = st.selectbox(
        "Choose Model", 
        options=list(model_options.keys()), 
        format_func=lambda x: model_options[x],
        index=0
    )
    
    # Temperature controls creativity vs factualness
    temperature = st.slider(
        "Creativity Level", 
        0.0, 1.0, 0.7, 
        help="Higher = More Creative, Lower = More Factual"
    )
    
    st.markdown("---")
    
    # The research topic - what the agents will investigate
    topic = st.text_input("Research Topic", "The Future of AI in 2026")
    
    st.markdown("---")
    
    # The launch button kicks everything off
    run_btn = st.button("Launch Research Crew", type="primary", use_container_width=True)


# This is where the magic happens when the user clicks launch
if run_btn:
    with st.status("**Agents are working...**", expanded=True) as status:
        try:
            st.markdown(f'{icon("search")} Researcher Agent is gathering data...', unsafe_allow_html=True)
            
            # Fire up the research crew with the right backend settings
            crew = ResearchCrew(
                topic=topic, 
                model_name=selected_model, 
                temperature=temperature,
                base_url=base_url,
                api_key=api_key
            )
            
            st.markdown(f'{icon("edit_note")} Writer Agent is drafting the content...', unsafe_allow_html=True)
            
            # Run the crew and get the final report
            result = crew.run()
            
            status.update(label="Research Complete!", state="complete", expanded=False)
            
            # Show the finished report
            st.markdown("---")
            st.markdown(f'<h3>{icon("article", "24px")} Report: {topic}</h3>', unsafe_allow_html=True)
            st.markdown(result)
            
            # Download button for saving the report
            st.download_button(
                label="Download Report",
                data=str(result),
                file_name=f"{topic.replace(' ', '_')}_report.md",
                mime="text/markdown"
            )

        except Exception as e:
            status.update(label="Error Occurred", state="error")
            st.error(f"An error occurred: {e}")
