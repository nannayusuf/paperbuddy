from __future__ import annotations
from crewai import Agent, Crew, Process, Task
from paperbuddy.tools.vector_search_tool import vector_search

def make_crew(user_query: str) -> Crew:
    tutor = Agent(
        role="Tutor de Artigos",
        goal="Responder dúvidas usando apenas evidências dos PDFs indexados",
        backstory="Especialista em leitura e síntese de artigos acadêmicos",
        tools=[vector_search],
        allow_delegation=False,
        verbose=True,
    )

    task = Task(
        description=f"Responder à pergunta do usuário usando apenas evidências dos artigos: {user_query}",
        agent=tutor,
        expected_output="Resposta clara com trechos relevantes encontrados nos artigos."
    )

    return Crew(agents=[tutor], tasks=[task], process=Process.sequential)
