import streamlit as st
from paperbuddy.crew import make_crew

st.set_page_config(page_title="PaperBuddy")
st.title("📚 PaperBuddy – Assistente de Estudos com CrewAI")

question = st.text_input("Qual sua dúvida sobre os PDFs?")
if st.button("Perguntar") and question:
    with st.spinner("Agentes trabalhando..."):
        crew = make_crew(question)
        result = crew.kickoff()
    st.success("Resposta do tutor:")
    st.write(result)