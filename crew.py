"""
Research Crew - The brain behind the operation.

This module sets up our AI research team using CrewAI. Think of it like 
assembling a small team of specialists: one person digs up the facts,
another person writes them up nicely. They work together in sequence
to turn your research topic into a polished report.

Now supports both local Ollama and cloud Groq inference - the app decides
which backend to use based on the environment.
"""

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI


class ResearchCrew:
    """
    Orchestrates a two-agent research workflow.
    
    The crew consists of a Senior Researcher (who gathers facts and data)
    and a Technical Writer (who turns those findings into readable content).
    Works with both local Ollama and cloud Groq - just pass the right URL and key.
    """
    
    def __init__(self, topic, model_name, temperature=0.7, base_url=None, api_key=None):
        """
        Sets up the crew with the given research topic and model preferences.
        
        Args:
            topic: What you want the agents to research (e.g., "quantum computing")
            model_name: Which model to use - could be local Ollama or cloud Groq
            temperature: How creative the responses should be (0.0 to 1.0)
            base_url: Where to send requests - localhost for Ollama, Groq URL for cloud
            api_key: API key for cloud providers, or "NA" for local Ollama
        """
        self.topic = topic
        
        # ChatOpenAI works with any OpenAI-compatible API, including Ollama and Groq.
        # The app.py decides which backend to use and passes us the right settings.
        self.llm = ChatOpenAI(
            model=model_name,
            base_url=base_url if base_url else "http://localhost:11434/v1",
            api_key=api_key if api_key else "NA",
            temperature=temperature
        )
    
    def run(self):
        """
        Kicks off the research crew and returns the final report.
        
        This method creates both agents, assigns them tasks, and runs the
        whole workflow from start to finish. The researcher goes first,
        then the writer takes those findings and creates the final output.
        
        Returns:
            The finished research report as a string (markdown formatted)
        """
        
        # Our researcher is the fact-finder. They're thorough and careful,
        # never making things up - just gathering solid information.
        researcher = Agent(
            role='Senior Research Analyst',
            goal=f'Find comprehensive and factual information about {self.topic}',
            backstory=(
                f"You are an incredibly skilled researcher with a deep understanding "
                f"of {self.topic}. Even with your expertise, you remain humble and "
                f"never make up information. You thoroughly fact-check all your findings "
                f"before presenting them."
            ),
            verbose=True,
            llm=self.llm 
        )

        # The writer takes raw research and turns it into something people
        # actually want to read. Professional but not boring.
        writer = Agent(
            role='Tech Content Strategist',
            goal=f"Summarize the researcher's findings into a clean, engaging post",
            backstory=(
                f"You are a talented technical writer who can take complex information "
                f"about {self.topic} and make it accessible to a broad audience. You "
                f"never change the facts - you just present them in a compelling way."
            ),
            verbose=True,
            llm=self.llm
        )

        # The research task tells our researcher exactly what to dig into
        research_task = Task(
            description=(
                f"Conduct a comprehensive analysis of {self.topic}. "
                f"Identify key trends, major players, and future predictions. "
                f"Focus on the most current and relevant data from 2025 and 2026."
            ),
            expected_output="A detailed bulleted report of findings.",
            agent=researcher
        )

        # The writing task transforms raw research into polished content
        writing_task = Task(
            description=(
                f"Using the research provided, write a high-impact blog post "
                f"about {self.topic}. The tone should be professional yet engaging. "
                f"Use Markdown formatting with proper headers (##) and emphasis (**bold**)."
            ),
            expected_output="A markdown-formatted blog post ready for publication.",
            agent=writer
        )
        
        # Assemble the crew and run tasks in sequence - researcher first, then writer
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            process=Process.sequential,
            verbose=True
        )

        # Kickoff returns the final output from the last task (the blog post)
        return crew.kickoff()
