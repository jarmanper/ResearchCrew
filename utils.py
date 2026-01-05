# Utility helpers for the Deep Research Agent

import re
import streamlit as st


class StreamToExpander:
    # Captures console output and shows it in a Streamlit code block
    # Strips out ANSI color codes and keeps only the last 15 lines
    
    def __init__(self, container):
        self.container = container
        self.buffer = []
        # Regex to match terminal color codes
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, data):
        if data.strip():
            clean_text = self.ansi_escape.sub('', data)
            self.buffer.append(clean_text)
            self.container.code("\n".join(self.buffer[-15:]), language="text")

    def flush(self):
        pass
