from crewai import Agent, Crew, Process, Task
from paperbuddy.tools.vector_search import vector_search_tool

coordinator = Agent(
    role="Coordinator",
    goal="Orquestrar análise multimodal de papers",
    backstory="Gestor que distribui tarefas entre agentes especializados"
)

text_agent = Agent(
    role="Text Analyst",
    goal="Extrair insights de texto do paper",
    tools=[vector_search_tool],
    backstory="Especialista em análise textual acadêmica"
)

vision_agent = Agent(
    role="Vision Analyst", 
    goal="Analisar figuras e tabelas",
    tools=[vector_search_tool],
    backstory="Expert em interpretação visual de papers"
)

rag_agent = Agent(
    role="RAG Synthesizer",
    goal="Sintetizar resposta final com contexto",
    tools=[vector_search_tool],
    backstory="Mestre em combinar texto e visão"
)

def process_query(query: str, paper_id: str) -> dict:
    tasks = [
        Task(description=f"Analisar texto: {query}", agent=text_agent),
        Task(description=f"Analisar figuras: {query}", agent=vision_agent),
        Task(description=f"Sintetizar resposta para: {query}", agent=rag_agent)
    ]
    
    crew = Crew(
        agents=[text_agent, vision_agent, rag_agent],
        tasks=tasks,
        process=Process.sequential
    )
    
    return crew.kickoff(inputs={"query": query, "paper_id": paper_id})