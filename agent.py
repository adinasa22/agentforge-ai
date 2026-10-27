import os
from typing import TypedDict

from openai import OpenAI


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4.1-mini"


# ---------------------------------------------------------
# Shared State
# ---------------------------------------------------------

class AgentState(TypedDict):
    user_request: str
    selected_agent: str
    agent_output: str
    review: str
    final_answer: str


# ---------------------------------------------------------
# LLM helper
# ---------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str) -> str:

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# Supervisor Agent
# ---------------------------------------------------------

def supervisor_agent(state: AgentState) -> AgentState:

    prompt = f"""
You are a supervisor agent.

Analyze the user's request and select the most appropriate specialist.

Available agents:

1. research
   - Research
   - Fact finding
   - Analysis
   - Comparisons

2. coding
   - Python
   - Programming
   - Debugging
   - Software architecture

3. general
   - Writing
   - Explanation
   - General questions

User request:

{state["user_request"]}

Return ONLY one of:

research
coding
general
"""

    selected = call_llm(
        "You are an AI task-routing supervisor.",
        prompt
    ).strip().lower()

    if selected not in ["research", "coding", "general"]:
        selected = "general"

    state["selected_agent"] = selected

    return state


# ---------------------------------------------------------
# Research Agent
# ---------------------------------------------------------

def research_agent(state: AgentState) -> AgentState:

    prompt = f"""
You are a research specialist.

Analyze the following request carefully:

{state["user_request"]}

Provide:

1. Key findings
2. Important assumptions
3. Technical reasoning
4. Recommended solution

Do not invent facts.
"""

    output = call_llm(
        "You are an expert research and analysis AI agent.",
        prompt
    )

    state["agent_output"] = output

    return state


# ---------------------------------------------------------
# Coding Agent
# ---------------------------------------------------------

def coding_agent(state: AgentState) -> AgentState:

    prompt = f"""
You are a senior Python engineer.

Solve the following request:

{state["user_request"]}

Requirements:

- Write clean Python
- Follow good software engineering practices
- Explain important parts
- Handle errors where appropriate
- Keep the implementation practical
"""

    output = call_llm(
        "You are an expert Python and AI engineering agent.",
        prompt
    )

    state["agent_output"] = output

    return state


# ---------------------------------------------------------
# General Agent
# ---------------------------------------------------------

def general_agent(state: AgentState) -> AgentState:

    prompt = f"""
Answer the following user request clearly and accurately:

{state["user_request"]}
"""

    output = call_llm(
        "You are a helpful general-purpose AI agent.",
        prompt
    )

    state["agent_output"] = output

    return state


# ---------------------------------------------------------
# Reviewer Agent
# ---------------------------------------------------------

def reviewer_agent(state: AgentState) -> AgentState:

    prompt = f"""
You are a senior AI reviewer.

Review the following proposed answer.

USER REQUEST:
{state["user_request"]}

PROPOSED ANSWER:
{state["agent_output"]}

Check for:

- Correctness
- Missing requirements
- Hallucinations
- Code problems
- Logical errors
- Clarity

Return:

APPROVED

if the answer is good.

Otherwise provide a corrected version.
"""

    review = call_llm(
        "You are a strict technical reviewer.",
        prompt
    )

    state["review"] = review

    return state


# ---------------------------------------------------------
# Finalizer Agent
# ---------------------------------------------------------

def finalizer_agent(state: AgentState) -> AgentState:

    prompt = f"""
Create the final answer to the user's request.

USER REQUEST:
{state["user_request"]}

ORIGINAL AGENT OUTPUT:
{state["agent_output"]}

REVIEW:
{state["review"]}

Produce a concise but useful final answer.

If the reviewer identified problems, fix them.
"""

    final_answer = call_llm(
        "You are the final response editor.",
        prompt
    )

    state["final_answer"] = final_answer

    return state


# ---------------------------------------------------------
# Multi-Agent Orchestrator
# ---------------------------------------------------------

def run_multi_agent_system(user_request: str):

    state: AgentState = {
        "user_request": user_request,
        "selected_agent": "",
        "agent_output": "",
        "review": "",
        "final_answer": ""
    }

    # Step 1: Supervisor decides
    state = supervisor_agent(state)

    print(f"\n[Supervisor] Selected: {state['selected_agent']}")

    # Step 2: Execute specialist
    if state["selected_agent"] == "research":
        state = research_agent(state)

    elif state["selected_agent"] == "coding":
        state = coding_agent(state)

    else:
        state = general_agent(state)

    print("[Specialist] Completed")

    # Step 3: Review
    state = reviewer_agent(state)

    print("[Reviewer] Completed")

    # Step 4: Final answer
    state = finalizer_agent(state)

    print("[Finalizer] Completed")

    return state["final_answer"]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    request = input("\nEnter your request: ")

    answer = run_multi_agent_system(request)

    print("\n================ FINAL ANSWER ================\n")
    print(answer)
