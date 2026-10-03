import os
import base64
import ast
import operator
from datetime import datetime
from typing import TypedDict, List, Annotated

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ============================================================
# ENVIRONMENT
# ============================================================
load_dotenv()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NMAMIT AI Academic Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# IMAGE LOADER
# ============================================================
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

bg_image = get_base64_image("assets/nitte.jpg")

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(f"""
<style>
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{background: transparent !important;}}
.stApp {{background: #f4f7fb;}}

/* ============ SIDEBAR ============ */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0b2542 0%, #123b66 100%);
    min-width: 240px !important;
    max-width: 240px !important;
    width: 240px !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    width: 240px !important;
    padding: 10px 12px !important;
}}
[data-testid="stSidebar"] * {{ color: #e8f0fa !important; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}}
.sidebar-logo {{
    padding: 6px 10px 11px 10px !important;
    margin: 0 0 7px 0 !important;
    border-bottom: 1px solid rgba(255,255,255,0.12);
}}
.sidebar-logo-title {{
    font-size: 18px !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.15 !important;
}}
.sidebar-logo-sub {{
    font-size: 10px !important;
    color: #a8c1dc !important;
    margin: 2px 0 0 0 !important;
    padding: 0 !important;
    line-height: 1.2 !important;
}}
.sidebar-section {{
    color: #91afd0 !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    padding: 11px 10px 4px 10px !important;
    margin: 0 !important;
    line-height: 1 !important;
    height: auto !important;
}}
[data-testid="stSidebar"] .stButton {{
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
}}
[data-testid="stSidebar"] .stButton > button {{
    width: 100% !important;
    height: 32px !important;
    min-height: 32px !important;
    max-height: 32px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    padding: 0 12px !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    color: #e8f0fa !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    line-height: 32px !important;
    text-align: left !important;
}}
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button div,
[data-testid="stSidebar"] .stButton > button span {{
    margin: 0 !important;
    padding: 0 !important;
    line-height: 32px !important;
    text-align: left !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255,255,255,0.10) !important;
}}

/* ============ HERO ============ */
.hero-banner {{
    position: relative;
    height: 285px;
    border-radius: 20px;
    overflow: hidden;
    background:
        linear-gradient(90deg, rgba(15,44,76,0.82), rgba(15,44,76,0.25)),
        url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(15,44,76,0.16);
}}
.hero-content {{
    position: absolute;
    left: 42px;
    top: 50%;
    transform: translateY(-50%);
    color: white;
    z-index: 2;
}}
.hero-small {{
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 1px;
    color: #dcecff !important;
    margin-bottom: 8px;
}}
.hero-title {{
    font-size: 43px;
    font-weight: 800;
    line-height: 1.08;
    color: #ffffff !important;
    margin: 0;
    text-shadow: 0 3px 12px rgba(0,0,0,0.35);
}}
.hero-subtitle {{
    font-size: 16px;
    line-height: 1.6;
    color: #edf5ff !important;
    max-width: 570px;
    margin-top: 12px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.30);
}}

/* ============ WELCOME ============ */
.welcome-section {{ padding: 0 4px 10px 4px; }}
.welcome-section h2 {{
    color: #0f2c4c;
    font-size: 25px;
    font-weight: 750;
    margin-bottom: 4px;
}}
.welcome-section p {{
    color: #607089;
    font-size: 14px;
    margin-top: 0;
}}

/* ============ MODE CHIP ============ */
.mode-chip {{
    display: inline-block;
    background: #e6f1ff;
    color: #1e6fd9;
    padding: 7px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 12px;
}}

/* ============ CHAT ============ */
.chat-divider {{
    height: 1px;
    background: #e5ebf2;
    margin: 20px 0;
}}
[data-testid="stChatMessage"] {{
    background: #ffffff !important;
    border-radius: 15px !important;
    padding: 12px 18px !important;
    margin-bottom: 10px !important;
    border: 1px solid #edf1f5;
    box-shadow: 0 3px 12px rgba(15,44,76,0.05);
}}

/* ============ CHAT INPUT ============ */
[data-testid="stChatInput"] {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
[data-testid="stChatInput"] > div {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
[data-testid="stChatInput"] textarea,
.stChatInput textarea {{
    background: transparent !important;
    border: 1px solid #d0d8e2 !important;
    border-radius: 28px !important;
    padding: 16px 23px !important;
    font-size: 15px !important;
    box-shadow: none !important;
    color: #1d3557 !important;
}}
[data-testid="stChatInput"] textarea::placeholder,
.stChatInput textarea::placeholder {{
    color: #71829a !important;
    opacity: 1 !important;
}}
[data-testid="stChatInput"] button,
.stChatInput button {{
    border-radius: 50% !important;
    background: #1e6fd9 !important;
    color: white !important;
    border: none !important;
}}

/* ============ QUICK QUESTIONS PANEL ============ */
.quick-title {{
    color: #0f2c4c;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 0.4px;
    margin: 4px 0 12px 0;
    text-transform: uppercase;
}}
.quick-sub {{
    color: #7a8699;
    font-size: 12px;
    margin: -6px 0 14px 0;
}}
.quick-panel .stButton > button {{
    width: 100% !important;
    text-align: left !important;
    background: #ffffff !important;
    color: #1d3557 !important;
    border: 1px solid #e3eaf2 !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    line-height: 1.35 !important;
    margin-bottom: 8px !important;
    box-shadow: 0 2px 8px rgba(15,44,76,0.04) !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 48px !important;
    transition: all 0.15s ease !important;
}}
.quick-panel .stButton > button:hover {{
    background: #f4f8ff !important;
    border-color: #b8d1ee !important;
    box-shadow: 0 4px 14px rgba(30,111,217,0.10) !important;
    transform: translateY(-1px);
}}
.quick-panel .stButton > button:focus {{
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(30,111,217,0.20) !important;
}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# VECTOR DATABASE
# ============================================================
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(persist_directory="chroma_db", embedding_function=embeddings)

try:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
except Exception:
    vectorstore = None
    retriever = None
    st.error("Knowledge base could not be loaded. Make sure your chroma_db folder exists.")

# ============================================================
# LLM
# ============================================================
try:
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
except Exception:
    llm = None

# ============================================================
# STATE
# ============================================================
class State(TypedDict):
    messages: Annotated[List, add_messages]
    question: str
    question_type: str
    retrieved_docs: List[str]
    draft_answer: str
    final_answer: str

# ============================================================
# MODE FOCUS & PLACEHOLDER
# ============================================================
MODE_FOCUS = {
    "Home": "",
    "Chat": "",
    "Syllabus": "Focus on NMAMIT syllabus, subjects, course codes, curriculum, credits, semesters and course structure.",
    "Examination": "Focus on examination rules, CIE, SEE, examination applications, hall tickets, valuation, results, revaluation and examination procedures.",
    "Academic Calendar": "Focus on academic calendar, semester dates, examination dates, holidays, registration dates and academic events.",
    "Study Planner": "Help the student create personalized academic study plans.",
    "Placement": "Focus on NMAMIT placement information, eligibility, recruitment process, companies, training, internships and placement statistics.",
    "Calculator": "Focus on mathematical calculations.",
    "Calendar": "Focus on dates, days, date differences and time-related questions.",
}

MODE_PLACEHOLDER = {
    "Home": "Ask anything about NMAMIT...",
    "Chat": "Type your question...",
    "Syllabus": "Ask about subjects, credits, or curriculum...",
    "Examination": "Ask about exams, CIE, SEE, results...",
    "Academic Calendar": "Ask about semester dates, holidays...",
    "Study Planner": "Describe your subjects, hours and exam date...",
    "Placement": "Ask about placements, companies, eligibility...",
    "Calculator": "Enter a mathematical expression...",
    "Calendar": "Ask about a date or how many days until...",
}

# ============================================================
# QUICK QUESTIONS
# ============================================================
QUICK_QUESTIONS = {
    "Home": [
        "What is the attendance requirement?",
        "What subjects are in 5th semester?",
        "How do I prepare for placements?",
        "What is the exam pattern?",
        "Make me a study plan for Math and Physics",
    ],
    "Chat": [
        "What is the attendance rule?",
        "Tell me about NMAMIT placements",
        "What clubs are available?",
        "How many days until 2026-12-15?",
        "What is 15% of 240?",
    ],
    "Syllabus": [
        "What is the total credit requirement?",
        "What subjects are in semester 1?",
        "What is CS2001-1?",
        "How many semesters in B.Tech?",
        "What are the professional electives?",
    ],
    "Examination": [
        "What is the minimum attendance for exams?",
        "What is CIE and SEE split?",
        "What is the malpractice process?",
        "How do I apply for retotalling?",
        "How many pages are in the UG answer booklet?",
    ],
    "Academic Calendar": [
        "When does the semester start?",
        "When are the SEE exams?",
        "When is the summer semester?",
        "How many days until 2026-12-15?",
        "What is today's date?",
    ],
    "Study Planner": [
        "Make me a study plan for Math, Physics, and English. Exam on 2026-12-15. 4 hours daily.",
        "Create a crash plan for CS and Math. Exam in 2 weeks. 5 hours daily.",
        "Plan for 5 subjects with 3 hours per day.",
        "Study plan for Data Structures and DBMS. Exam on 2026-11-20.",
        "Weekly revision plan for 3 subjects.",
    ],
    "Placement": [
        "What is the placement department called?",
        "How many companies visited in 2023?",
        "What is the highest salary offered?",
        "Which Japanese companies hire from NMAMIT?",
        "How many CSE students were placed in 2026?",
    ],
    "Calculator": [
        "What is 15% of 240?",
        "What is 125 * 8 + 50?",
        "Calculate 500 / 4",
        "What is 20% of 75?",
        "What is 2 to the power of 10?",
    ],
    "Calendar": [
        "What is today's date?",
        "How many days until 2026-12-15?",
        "How many days until January 1, 2027?",
        "What time is it now?",
        "What day is it today?",
    ],
}

# ============================================================
# NODE 1 - ANALYZE (FIXED — rule-based overrides)
# ============================================================
def analyze_question(state: State) -> State:
    question = state["messages"][-1].content
    question_lower = question.lower()

    # ---- RULE: Planner detection ----
    planner_keywords = [
        "study plan", "study planner", "schedule", "timetable",
        "exam preparation", "prepare for exam", "crash plan",
        "revision plan", "revision schedule", "make me a plan",
        "help me plan", "plan for", "study routine",
        "plan my", "plan for my exams", "plan for exam",
        "how to plan", "make a plan", "create a plan",
        "study schedule", "exam plan",
    ]
    if any(kw in question_lower for kw in planner_keywords):
        return {"question": question, "question_type": "planner"}

    # ---- RULE: Math detection ----
    if any(ch in question for ch in ["+", "*", "/"]) and any(c.isdigit() for c in question):
        return {"question": question, "question_type": "math"}
    if "percent" in question_lower and any(c.isdigit() for c in question):
        return {"question": question, "question_type": "math"}

    # ---- RULE: Date detection ----
    if ("how many days" in question_lower
            or "days until" in question_lower
            or "days left" in question_lower
            or "days remain" in question_lower):
        return {"question": question, "question_type": "date"}

    # ---- RULE: Time detection ----
    if ("today's date" in question_lower
            or "what is the date" in question_lower
            or "what time" in question_lower
            or "current time" in question_lower
            or "what day is it" in question_lower):
        return {"question": question, "question_type": "time"}

    # ---- Fall back to LLM for RAG / unknown ----
    if llm is None:
        return {"question": question, "question_type": "rag"}

    prompt = f"""Classify the student's question into ONE category.

Categories:
rag      → any NMAMIT / college academic question
unknown  → unrelated question

Question: {question}

Reply with ONLY one word: rag or unknown"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        category = response.content.strip().lower().split()[0]
    except Exception:
        category = "rag"

    if category not in ["rag", "unknown"]:
        category = "rag"

    return {"question": question, "question_type": category}

# ============================================================
# SAFE CALCULATOR
# ============================================================
allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def safe_calculate(expression):
    def calculate(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Invalid number")
        if isinstance(node, ast.BinOp):
            left = calculate(node.left)
            right = calculate(node.right)
            operation = allowed_operators.get(type(node.op))
            if operation is None:
                raise ValueError("Operator not allowed")
            return operation(left, right)
        if isinstance(node, ast.UnaryOp):
            operation = allowed_operators.get(type(node.op))
            if operation is None:
                raise ValueError("Operator not allowed")
            return operation(calculate(node.operand))
        raise ValueError("Invalid expression")
    tree = ast.parse(expression, mode="eval")
    return calculate(tree.body)

# ============================================================
# NODE 2 - RETRIEVE / TOOL
# ============================================================
def retrieve_or_tool(state: State) -> State:
    qtype = state["question_type"]
    question = state["question"]

    if qtype == "rag":
        if retriever is None:
            return {"retrieved_docs": []}
        try:
            docs = retriever.invoke(question)
            return {"retrieved_docs": [d.page_content for d in docs]}
        except Exception:
            return {"retrieved_docs": []}

    elif qtype == "math":
        if llm is None:
            return {"retrieved_docs": ["Calculator unavailable."]}
        try:
            expression_response = llm.invoke([HumanMessage(content=f"""Extract ONLY the mathematical expression from this question.

Question: {question}

Reply with ONLY the expression.

Examples:
"What is 25 * 8?" → 25 * 8
"Calculate 20 percent of 500" → 500 * 20 / 100""")])
            expression = expression_response.content.strip()
            result = safe_calculate(expression)
            return {"retrieved_docs": [f"Calculated: {expression} = {result}"]}
        except Exception as e:
            return {"retrieved_docs": [f"Calculator error: {e}"]}

    elif qtype == "date":
        if llm is None:
            return {"retrieved_docs": ["Date tool unavailable."]}
        today = datetime.now()
        try:
            response = llm.invoke([HumanMessage(content=f"""Extract the target date as YYYY-MM-DD.

Today is: {today.strftime('%Y-%m-%d')}

Question: {question}

Reply with ONLY the date.""")])
            target = response.content.strip()
            target_dt = datetime.strptime(target, "%Y-%m-%d")
            delta = (target_dt.date() - today.date()).days
            if delta > 0:
                message = f"{delta} days remain until {target}."
            elif delta == 0:
                message = f"{target} is today."
            else:
                message = f"{target} was {abs(delta)} days ago."
            return {"retrieved_docs": [message]}
        except Exception:
            return {"retrieved_docs": ["Could not understand the requested date."]}

    elif qtype == "time":
        now = datetime.now()
        return {"retrieved_docs": [now.strftime("Today is %A, %B %d, %Y. Time: %I:%M %p.")]}

    elif qtype == "planner":
        return {"retrieved_docs": ["__PLANNER__"]}

    return {"retrieved_docs": []}

# ============================================================
# NODE 3 - GENERATE RESPONSE
# ============================================================
def generate_response(state: State) -> State:
    question = state["question"]
    context = "\n\n".join(state["retrieved_docs"]) if state["retrieved_docs"] else ""
    qtype = state["question_type"]
    mode = st.session_state.get("mode", "Home")
    mode_focus = MODE_FOCUS.get(mode, "")

    if qtype in ["math", "date", "time"]:
        prompt = f"""Tool result: {context}

Question: {question}

Give a short and clear answer based ONLY on the tool result."""
        if llm is None:
            answer = context
        else:
            answer = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"draft_answer": answer}

    if qtype == "unknown":
        return {"draft_answer": "I don't have that information in my knowledge base."}

    # ============================================================
    # PLANNER — bulletproof prompt (must NOT ask questions)
    # ============================================================
    if qtype == "planner":
        today = datetime.now()
        today_str = today.strftime("%A, %B %d, %Y")

        prompt = f"""=== MANDATORY INSTRUCTION ===
You are a study planner. You MUST produce a complete study plan NOW.
You are FORBIDDEN from asking the student any questions.
You are FORBIDDEN from requesting more information.
You MUST make reasonable assumptions and write the plan.
If you ask a question instead of producing a plan, you FAIL your task.
=== END MANDATORY INSTRUCTION ===

Today's date: {today_str}

Student's request:
"{question}"

Produce the study plan RIGHT NOW in this exact structure:

## 📅 Study Plan Overview
- Exam date: (extract from request, or assume 60 days from today)
- Days remaining: (calculate from today's date)
- Daily study hours: (as stated, or assume 4)
- Subjects: (as listed, or assume 3 typical B.Tech subjects)

## 📆 Weekly Schedule
| Day | Subject 1 | Subject 2 | Subject 3 | Revision |
|-----|-----------|-----------|-----------|----------|
| Monday | ... | ... | ... | ... |
| Tuesday | ... | ... | ... | ... |
| Wednesday | ... | ... | ... | ... |
| Thursday | ... | ... | ... | ... |
| Friday | ... | ... | ... | ... |
| Saturday | ... | ... | ... | ... |
| Sunday | Rest + light revision | | | |

## ⏰ Daily Time Allocation
- Split the daily hours across subjects with specific times (e.g., "6-7 PM: Math")
- Include 10-minute breaks every hour

## 📊 Subject-wise Weekly Hours
| Subject | Hours/Week |
|---------|-----------|
| Subject 1 | X hrs |
| Subject 2 | X hrs |
| Subject 3 | X hrs |

## 🎯 Revision Strategy
- Weekly revision day
- Mock test schedule for the final 2 weeks
- Progress tracking method

## 💡 Study Tips
1. Tip 1
2. Tip 2
3. Tip 3
4. Tip 4
5. Tip 5

## 📝 Final 2 Weeks Plan
- Intensive revision mode
- Mock tests every alternate day
- Focus on weak areas

REMEMBER: Do NOT ask any questions. Do NOT request clarification. Write the plan immediately."""

        if llm is None:
            answer = "Study planner is currently unavailable."
        else:
            answer = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"draft_answer": answer}

    # ============================================================
    # RAG — typo-tolerant prompt
    # ============================================================
    if not context.strip():
        return {"draft_answer": "I don't have that information in my knowledge base."}

    prompt = f"""You are NMAMIT's AI Academic Assistant.

CURRENT MODE: {mode}
MODE FOCUS: {mode_focus}

Answer the student's question using the provided CONTEXT.

CRITICAL RULES:
1. The student may have TYPOS, missing letters, or broken English. ALWAYS understand the intent first.
   - "hwo to gt honours degre" = "How to get honours degree?"
   - "wat is da atendence rule" = "What is the attendance rule?"
   - "I'm a 5th sem CSE student. What subjects do I have?" = "List semester 5 CSE subjects"
2. Use the CONTEXT as the primary source. If the CONTEXT contains information that could reasonably answer the question (even partially), USE IT.
3. IMPORTANT: Even if only PART of the answer is in the context, provide what IS there and note any gaps.
4. Only refuse if the CONTEXT is COMPLETELY unrelated to the question.
5. If you must refuse, say exactly: "I don't have that information in my knowledge base."
6. Give a direct answer first, then details.
7. Use headings and bullet points for readability.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    if llm is None:
        answer = "LLM is not available. Please check your GROQ API key."
    else:
        answer = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    return {"draft_answer": answer}

# ============================================================
# NODE 4 - REVIEW
# ============================================================
def review_response(state: State) -> State:
    draft = state["draft_answer"]
    if not draft:
        draft = "I don't have that information in my knowledge base."
    return {"final_answer": draft}

# ============================================================
# BUILD LANGGRAPH
# ============================================================
builder = StateGraph(State)
builder.add_node("analyze", analyze_question)
builder.add_node("retrieve", retrieve_or_tool)
builder.add_node("generate", generate_response)
builder.add_node("review", review_response)
builder.add_edge(START, "analyze")
builder.add_edge("analyze", "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "review")
builder.add_edge("review", END)
graph = builder.compile()

# ============================================================
# SESSION STATE
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mode" not in st.session_state:
    st.session_state.mode = "Home"
if "mode_focus" not in st.session_state:
    st.session_state.mode_focus = ""
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ============================================================
# HELPERS
# ============================================================
def change_mode(mode):
    st.session_state.mode = mode
    st.session_state.mode_focus = MODE_FOCUS.get(mode, "")
    st.session_state.chat_history = []
    st.session_state.pending_question = None
    st.rerun()

def run_query(question):
    try:
        result = graph.invoke({"messages": [HumanMessage(content=question)]})
        return result["final_answer"]
    except Exception as e:
        return f"Something went wrong while processing your question.\n\nError: {e}"

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <p class="sidebar-logo-title">🎓 NMAMIT</p>
        <p class="sidebar-logo-sub">AI Academic Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Main</div>', unsafe_allow_html=True)
    for label, mode in [("Home", "Home"), ("Chat", "Chat")]:
        if st.button(label, key=f"main_{mode}", use_container_width=True):
            change_mode(mode)

    st.markdown('<div class="sidebar-section">Academic</div>', unsafe_allow_html=True)
    for label, mode in [
        ("Syllabus", "Syllabus"),
        ("Examination", "Examination"),
        ("Academic Calendar", "Academic Calendar"),
        ("Study Planner", "Study Planner"),
        ("Placement", "Placement"),
    ]:
        if st.button(label, key=f"acad_{mode}", use_container_width=True):
            change_mode(mode)

    st.markdown('<div class="sidebar-section">Tools</div>', unsafe_allow_html=True)
    for label, mode in [("Calculator", "Calculator"), ("Calendar", "Calendar")]:
        if st.button(label, key=f"tool_{mode}", use_container_width=True):
            change_mode(mode)

# ============================================================
# MAIN AREA + RIGHT SIDEBAR
# ============================================================
main_col, quick_col = st.columns([3, 1], gap="large")

with main_col:
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-content">
            <div class="hero-small">🎓 NMAMIT</div>
            <h1 class="hero-title">AI Academic Assistant</h1>
            <p class="hero-subtitle">Your intelligent companion for academics, examinations, study plans and campus life at NMAMIT.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.mode != "Home":
        st.markdown(f'<div class="mode-chip">🔎 {st.session_state.mode} Assistant</div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        if st.session_state.mode == "Home":
            st.markdown("""
            <div class="welcome-section">
                <h2>How can I help you today?</h2>
                <p>Ask me anything about NMAMIT academics, examinations, syllabus, placements and more.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="welcome-section">
                <h2>{st.session_state.mode}</h2>
                <p>Ask your questions related to {st.session_state.mode.lower()}.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="chat-divider"></div>', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    placeholder = MODE_PLACEHOLDER.get(st.session_state.mode, "Ask anything about NMAMIT...")
    user_input = st.chat_input(placeholder)

    if st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "⏳ Thinking..."
        })
        st.rerun()

with quick_col:
    st.markdown('<div class="quick-title">⚡ Quick Questions</div>', unsafe_allow_html=True)
    st.markdown('<div class="quick-sub">Click a question to ask instantly</div>', unsafe_allow_html=True)

    current_mode = st.session_state.mode
    questions = QUICK_QUESTIONS.get(current_mode, QUICK_QUESTIONS["Home"])

    st.markdown('<div class="quick-panel">', unsafe_allow_html=True)
    for i, q in enumerate(questions):
        if st.button(q, key=f"quick_{current_mode}_{i}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="quick-sub" style="margin-top:18px;">💡 Questions change based on the mode you select on the left.</div>',
        unsafe_allow_html=True
    )

# ============================================================
# PROCESS PENDING AI RESPONSE
# ============================================================
if st.session_state.chat_history and st.session_state.chat_history[-1]["content"] == "⏳ Thinking...":
    user_question = None
    for msg in reversed(st.session_state.chat_history[:-1]):
        if msg["role"] == "user":
            user_question = msg["content"]
            break

    if user_question:
        answer = run_query(user_question)
        st.session_state.chat_history[-1]["content"] = answer
        st.rerun()