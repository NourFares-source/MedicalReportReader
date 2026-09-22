from dotenv import load_dotenv
import os
import base64
import pdfplumber
import subprocess
from typing import List, Optional, TypedDict
from PIL import Image
import io
import streamlit as st
import os
# LangChain & Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# LangGraph
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# Microsoft Presidio
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer.entities import OperatorConfig

# Colab / Streamlit helpers


load_dotenv()  # Load environment variables from .env file
class MedicalReportState(TypedDict):
    # Inputs
    user_intent: str
    file_bytes: bytes
    file_type: str  # 'pdf' or 'image'

    # Processed Data
    redacted_text: str

    # AI Generation
    analysis_result: str
    audit_feedback: Optional[str]
    iteration_count: int

    # Security Flag
    is_safe: bool



def security_node(state: MedicalReportState):
    # Initialize Presidio
    # Configure Presidio to explicitly use the small spaCy model
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    anonymizer = AnonymizerEngine()

    # 1. Extraction (Simplified for this node)
    # We will build the robust extractor in the Collector node,
    # but for now, we assume we have text.
    text_to_scrub = state.get("redacted_text", "")

    # 2. Analyze the text for PII
    results = analyzer.analyze(text=text_to_scrub, language='en',
                               entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"])

    # 3. Define the "Redacted" Operator
    operators = {
        "PERSON": OperatorConfig("replace", {"new_value": "[REDACTED_NAME]"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[REDACTED_PHONE]"}),
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[REDACTED_EMAIL]"}),
        "LOCATION": OperatorConfig("replace", {"new_value": "[REDACTED_LOCATION]"}),
    }

    # 4. Execute Anonymization
    anonymized_result = anonymizer.anonymize(
        text=text_to_scrub,
        analyzer_results=results,
        operators=operators
    )

    #print("🛡️ Security Node: PII Scrutiny Complete.")
    return {"redacted_text": anonymized_result.text, "is_safe": True}


def collector_node(state: MedicalReportState):
    file_bytes = state["file_bytes"]
    file_type = state["file_type"]
    extracted_text = ""

    print(f"📥 Collector Node: Processing {file_type}...")

    # --- STRATEGY A: PDF EXTRACTION ---
    if file_type == "pdf":
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages_text = []
                for page in pdf.pages:
                    # Extracting text while preserving layout
                    text = page.extract_text(layout=True)
                    if text:
                        pages_text.append(text)
                extracted_text = "\n\n".join(pages_text)
        except Exception as e:
            print(f"❌ PDF Error: {e}")

    # --- STRATEGY B: IMAGE OCR (Gemini) ---
    elif file_type in ["image", "png", "jpg", "jpeg"]:
        llm_vision = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )

        # Convert bytes to base64 for Gemini
        img_b64 = base64.b64encode(file_bytes).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": "Extract all medical data from this image into a clean markdown table format. Include test names, results, units, and reference ranges."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        )
        res = llm_vision.invoke([message])
        extracted_text = res.content

    if not extracted_text:
        print("⚠️ Warning: No text was extracted.")

    return {"redacted_text": extracted_text} # Passing to Security Node next



def analysis_node(state: MedicalReportState):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2 # Lower temperature for factual accuracy
    )

    intent = state["user_intent"]
    report_data = state["redacted_text"]

    prompt = f"""
    You are a professional medical report analyst.
    The user's primary concern/intent is: {intent}.

    TASK:
    1. Scan the following redacted blood test report: {report_data}
    2. Identify all values that are outside the 'Reference Range'.
    3. Specifically prioritize markers related to the intent: {intent}.
    4. For each abnormal value, provide:
       - The name of the test.
       - Why it might be high/low.
       - A lifestyle or dietary suggestion.
       - The specific type of specialist doctor they should consult.

    GUIDELINES:
    - Maintain a professional, empathetic tone.
    - Be extremely precise with the numbers.
    - DO NOT diagnose a disease; use phrases like 'Your levels suggest...' or 'It is recommended to discuss X with a doctor.'
    """

    res = llm.invoke([HumanMessage(content=prompt)])

    print("🧠 Analysis Node: Medical Reasoning Complete.")
    return {"analysis_result": res.content, "iteration_count": state.get("iteration_count", 0) + 1}


def audit_node(state: MedicalReportState):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Using the latest model for high-precision auditing
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.0 # ZERO creativity allowed here
    )

    report_data = state["redacted_text"]
    analysis = state["analysis_result"]

    prompt = f"""
    You are a Medical Data Auditor. Your job is to verify the accuracy of a medical analysis.

    ORIGINAL DATA:
    {report_data}

    GENERATED ANALYSIS:
    {analysis}

    VERIFICATION TASKS:
    1. Check the numbers: Did the analysis accurately reflect the values in the original data?
    2. Check the ranges: Did the analysis correctly identify if a value was high, low, or normal based on the lab's reference ranges?
    3. Check for Hallucinations: Did the analysis mention any tests or results that ARE NOT in the original data?

    RESPONSE FORMAT:
    If everything is 100% accurate, respond with 'PASS'.
    If there is an error, respond with 'FAIL' followed by a detailed list of corrections.
    """

    res = llm.invoke([HumanMessage(content=prompt)])
    audit_output = res.content.strip()

    if "PASS" in audit_output.upper():
        print("✅ Audit Node: Analysis Verified.")
        return {"audit_feedback": None, "is_safe": True}
    else:
        print("🚨 Audit Node: Hallucination/Error Detected!")
        return {"audit_feedback": audit_output, "is_safe": False}


workflow = StateGraph(MedicalReportState)

# 1. Add the Nodes
workflow.add_node("collector", collector_node)
workflow.add_node("security", security_node)
workflow.add_node("analysis", analysis_node)
workflow.add_node("audit", audit_node)

# 2. Define the Edges
workflow.add_edge(START, "collector")
workflow.add_edge("collector", "security")
workflow.add_edge("security", "analysis")
workflow.add_edge("analysis", "audit")

# 3. Add the Conditional Logic (The Loop)
def should_continue(state: MedicalReportState):
    if state["is_safe"] or state.get("iteration_count", 0) >= 3:
        return END
    return "analysis"

workflow.add_conditional_edges("audit", should_continue)

# 4. Compile
app = workflow.compile()
print("🕸️ Graph Compiled Successfully!")


st.set_page_config(page_title="MediScan AI", page_icon="🩸", layout="centered")

st.title("🩸 MediScan: Intelligent Report Reader")
st.markdown("---")

# 1. Sidebar for Intent and Disclaimer
with st.sidebar:
    st.header("Settings")
    intent = st.selectbox(
        "What is the primary reason for this test?",
        ["General Health Check", "Energy & Fatigue", "Heart & Cholesterol", "Diabetes/Sugar Check", "Dietary/Vegan Check"]
    )
    st.warning("**Disclaimer:** This AI is for educational purposes only. Always consult a doctor for diagnosis.")

# 2. File Upload
uploaded_file = st.file_uploader("Upload your blood test (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    with st.spinner("Processing report... (Redacting PII & Analyzing)"):
        # Prepare inputs for the graph
        file_bytes = uploaded_file.getvalue()
        file_type = uploaded_file.name.split('.')[-1].lower()

        # Invoke the LangGraph App
        inputs = {
            "file_bytes": file_bytes,
            "file_type": file_type,
            "user_intent": intent,
            "iteration_count": 0
        }

        result = app.invoke(inputs)

        # 3. Display Results
        st.success("Analysis Complete!")

        tab1, tab2 = st.tabs(["📋 Medical Analysis", "🛡️ Privacy Check (Redacted Text)"])

        with tab1:
            st.subheader("Key Findings & Recommendations")
            st.markdown(result.get("analysis_result", "No analysis generated."))

        with tab2:
            st.info("Below is the text the AI saw. Notice how your personal details were removed.")
            st.code(result.get("redacted_text", "No text found."))

st.markdown("---")
st.caption("Powered by LangGraph, Gemini 2.5, and Microsoft Presidio.")
