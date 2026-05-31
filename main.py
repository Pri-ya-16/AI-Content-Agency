# =========================================================
# AI CONTENT AGENCY SWARM UI
# STREAMLIT + CREWAI + GROQ
# =========================================================

# =========================================================
# IMPORTS
# =========================================================

import os
import time

import streamlit as st

from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Content Agency",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🤖 AI Content Agency Swarm")
st.subheader("Multi-Agent Content Generation ")

# =========================================================
# GROQ LLM
# =========================================================

llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5,
    max_tokens=500
)

# =========================================================
# USER INPUT
# =========================================================

topic = st.text_input(
    "Enter Your Topic",
    placeholder="Enter a topic for the content?"
)

# =========================================================
# GENERATE BUTTON
# =========================================================

if st.button("Generate Content"):

    if topic.strip() == "":
        st.warning("Please enter a topic.")
        st.stop()

    # =====================================================
    # LOADING
    # =====================================================

    with st.spinner("Agents are working on your content..."):

        # =================================================
        # MANAGER AGENT
        # =================================================

        manager = Agent(
            role="Manager",

            goal="Manage content workflow.",

            backstory="You manage AI content teams.",

            verbose=True,

            llm=llm
        )

        # =================================================
        # RESEARCH AGENT
        # =================================================

        researcher = Agent(
            role="Research Agent",

            goal="Research the topic and gather insights.",

            backstory="You are a professional researcher.",

            verbose=True,

            llm=llm
        )

        # =================================================
        # WRITER AGENT
        # =================================================

        writer = Agent(
            role="Writer Agent",

            goal="Write professional blog content.",

            backstory="You are an SEO blog writer.",

            verbose=True,

            llm=llm
        )

        # =================================================
        # EDITOR AGENT
        # =================================================

        editor = Agent(
            role="Editor Agent",

            goal="Improve readability and SEO.",

            backstory="You are a senior editor.",

            verbose=True,

            llm=llm
        )

        # =================================================
        # TASKS
        # =================================================

        research_task = Task(
            description=f"""
            Research topic: {topic}

            Give:
            - 5 key points
            - trends
            - short notes
            """,

            expected_output="Short research notes.",

            agent=researcher
        )

        writing_task = Task(
            description=f"""
            Write a short blog article on:

            {topic}

            Include:
            - Title
            - Intro
            - 3 headings
            - Conclusion
            """,

            expected_output="SEO-friendly blog article.",

            agent=writer
        )

        editing_task = Task(
            description="""
            Improve grammar, readability,
            and SEO formatting.
            """,

            expected_output="Final polished article.",

            agent=editor
        )

        # =================================================
        # CREATE CREW
        # =================================================

        crew = Crew(
            agents=[
                manager,
                researcher,
                writer,
                editor
            ],

            tasks=[
                research_task,
                writing_task,
                editing_task
            ],

            process=Process.sequential,

            verbose=False
        )

        # =================================================
        # EXECUTION
        # =================================================

        time.sleep(2)

        result = crew.kickoff()

    # =====================================================
    # OUTPUT
    # =====================================================

    st.success("Content Generated Successfully!")

    st.markdown("---")

    st.markdown(result)