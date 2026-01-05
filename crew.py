# Research Crew
# Sets up the AI research team using CrewAI
# One agent digs up facts, another writes them up nicely

from crewai import Agent, Task, Crew, Process, LLM


class ResearchCrew:
    # Runs a two-agent workflow: researcher finds info, writer makes it readable
    # Works with both local Ollama and cloud Groq
    
    def __init__(self, topic, model_name, temperature=0.7, api_key=None):
        self.topic = topic
        
        # CrewAI's LLM class uses litellm under the hood
        # Model names include the provider prefix (groq/, ollama/) for routing
        self.llm = LLM(
            model=model_name,
            api_key=api_key,
            temperature=temperature
        )
    
    def run(self):
        # Create the researcher agent - finds facts, never makes stuff up
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

        # Create the writer agent - takes research and makes it readable
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

        # Define what the researcher should focus on
        research_task = Task(
            description=(
                f"Conduct a comprehensive analysis of {self.topic}. "
                f"Identify key trends, major players, and future predictions. "
                f"Focus on the most current and relevant data from 2025 and 2026."
            ),
            expected_output="A detailed bulleted report of findings.",
            agent=researcher
        )

        # Define what the writer should produce
        writing_task = Task(
            description=(
                f"Using the research provided, write a high-impact blog post "
                f"about {self.topic}. The tone should be professional yet engaging. "
                f"Use Markdown formatting with proper headers (##) and emphasis (**bold**)."
            ),
            expected_output="A markdown-formatted blog post ready for publication.",
            agent=writer
        )
        
        # Put it all together and run in sequence
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            process=Process.sequential,
            verbose=True
        )

        return crew.kickoff()
