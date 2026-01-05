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
        # Create the researcher agent - thorough, detailed, fact-focused
        researcher = Agent(
            role='Senior Research Analyst',
            goal=f'Produce an extremely detailed and comprehensive research brief on {self.topic}',
            backstory=(
                f"You are a world-class research analyst known for producing thorough, "
                f"well-structured reports. You dig deep into topics, exploring multiple angles, "
                f"statistics, examples, and expert opinions. You never skim the surface - you "
                f"provide substantive analysis with specific details, data points, and concrete examples. "
                f"You cite sources when possible and distinguish between established facts and emerging trends."
            ),
            verbose=True,
            llm=self.llm 
        )

        # Create the writer agent - transforms research into polished content
        writer = Agent(
            role='Tech Content Strategist',
            goal=f"Transform research into a comprehensive, well-organized article that maintains depth while being readable",
            backstory=(
                f"You are an expert technical writer who excels at making complex topics accessible "
                f"without dumbing them down. You preserve important details and nuances from research "
                f"while organizing them into clear sections. You use specific examples, statistics, "
                f"and concrete details - never vague generalizations. Your writing is information-dense "
                f"but scannable, with clear headers and logical flow."
            ),
            verbose=True,
            llm=self.llm
        )

        # Research task - push for depth and specifics
        research_task = Task(
            description=(
                f"Conduct an in-depth analysis of {self.topic}. Your research must include:\n"
                f"1. Background and context - what is this and why does it matter?\n"
                f"2. Current state - what's happening right now? Include specific examples, companies, or projects.\n"
                f"3. Key statistics and data points - numbers, percentages, growth rates.\n"
                f"4. Major players and stakeholders - who's involved and what are they doing?\n"
                f"5. Challenges and controversies - what are the problems or debates?\n"
                f"6. Future outlook - where is this heading in the next 1-3 years?\n"
                f"7. Expert opinions or notable quotes if available.\n\n"
                f"Be specific. Use concrete examples, not vague statements. "
                f"Focus on information from 2024-2026."
            ),
            expected_output=(
                "A comprehensive research brief with 7+ distinct sections, "
                "including specific examples, statistics, and detailed analysis. "
                "At least 800 words of substantive content."
            ),
            agent=researcher
        )

        # Writing task - maintain depth while improving readability
        writing_task = Task(
            description=(
                f"Using the research provided, create a comprehensive article about {self.topic}. "
                f"Requirements:\n"
                f"1. Keep ALL the specific details, statistics, and examples from the research.\n"
                f"2. Organize into clear sections with descriptive headers (use ## for headers).\n"
                f"3. Start with a compelling introduction that frames why this matters.\n"
                f"4. Use bullet points for lists of items or key points.\n"
                f"5. Bold (**) important terms, statistics, or key findings.\n"
                f"6. End with a forward-looking conclusion.\n"
                f"7. Maintain an authoritative but accessible tone.\n\n"
                f"Do NOT summarize or condense - preserve the depth of the research."
            ),
            expected_output=(
                "A polished, publication-ready article in Markdown format. "
                "Should be 800-1200 words with clear structure, preserved details, "
                "and professional formatting."
            ),
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
