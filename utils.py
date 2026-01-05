"""
Utility helpers for the Deep Research Agent.

This module contains helper classes that make the app work more smoothly.
Right now it's just the stream capture utility, but this is a good place
to add any other shared functionality as the project grows.
"""

import re
import streamlit as st


class StreamToExpander:
    """
    Captures console output and displays it in a Streamlit expander.
    
    When the AI agents are working, they print a lot of stuff to the console.
    This class intercepts that output and shows it in a nice code block inside
    the Streamlit app. It keeps only the last 15 lines so things don't get 
    too cluttered.
    
    The class also strips out ANSI escape codes (those weird characters that
    make terminal text colorful) so the output looks clean in the browser.
    """
    
    def __init__(self, container):
        """
        Sets up the stream capturer.
        
        Args:
            container: A Streamlit container element where we'll display the output
        """
        self.container = container
        self.buffer = []  # Holds the lines of captured output
        
        # This regex matches ANSI escape sequences - the invisible codes terminals
        # use for colors and formatting. We strip these out since they just look
        # like garbage in a web browser.
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, data):
        """
        Called whenever something tries to print to this stream.
        
        We clean up the text, add it to our buffer, and update the display.
        Only non-empty lines get added - no point showing blank lines.
        
        Args:
            data: The string that was printed to the stream
        """
        if data.strip():
            # Remove any terminal color codes before displaying
            clean_text = self.ansi_escape.sub('', data)
            self.buffer.append(clean_text)
            
            # Show only the most recent 15 lines to keep things readable
            # Using a code block gives us nice monospace formatting
            self.container.code("\n".join(self.buffer[-15:]), language="text")

    def flush(self):
        """
        Required for file-like objects but we don't need to do anything here.
        
        Python expects streams to have a flush method, so we provide one
        even though our buffer updates are immediate and don't need flushing.
        """
        pass
