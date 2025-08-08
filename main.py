# ==============================================================================
# 1. Dependencies & Environment Setup
# ==============================================================================
# Standard Libraries
import json
import uuid
from typing import List, Dict, TypedDict, Optional

# Third-Party Libraries
import feedparser
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain & LangGraph Core Components
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

# --- Global Constants Definition ---
# Maximum number of attempts for the auto-revision loop before triggering a fallback.
MAX_FAILS = 3
# To prevent infinite loops, set a total limit on rewrites.
MAX_TOTAL = 10


# ==============================================================================
# 2. Pydantic Models
# ==============================================================================
# This section defines the structured data models that the LLM needs to generate at different stages.

class RequestPreprocessModel(BaseModel):
    """A structured model for parsing the user's initial request."""
    industry: str = Field(..., description="The industry or domain mentioned in the user's request.")
    language: str = Field(..., description="The desired output language, using language codes (e.g., en, de, zh).")
    country_code: Optional[str] = Field(None, description="The country for which the user wants to see trends, using country codes (e.g., US, DE). If not specified, it's inferred from the language.")
    keywords: List[str] = Field(description="3 to 5 keywords or phrases optimized for a Wikipedia search.")


class RelevantTrends(BaseModel):
    """A model for filtering relevant trend titles from a Google Trends list."""
    trend_titles: List[str] = Field(
        ...,
        description="A list of trend titles that are directly relevant to the user's request. All titles must come exactly from the provided data list."
    )


class Critiques(BaseModel):
    """A structured model for critiquing the generated draft article."""
    critique_list: List[str] = Field(
        ...,
        description="A list of actionable, specific revision suggestions. If the draft is of high quality and needs no changes, return an empty list."
    )

# Defines the default mapping from language to country code.
LANGUAGE_TO_COUNTRY = {"en": "US", "de": "DE", "fr": "FR", "zh": "CN"}


# ==============================================================================
# 3. Agent State Definition
# ==============================================================================

class AgentState(TypedDict):
    """
    Defines the state that is passed and shared throughout the entire workflow.
    This state object acts as the agent's "memory," which each node can read from and write to.
    """
    # --- Initial Input ---
    request: str                  # The user's original request string.

    # --- Preprocessing Stage Output ---
    industry: str                 # The industry extracted from the request.
    language: str                 # The target language code.
    country_code: str             # The target country code.
    keywords: List[str]           # Keywords for fallback search.

    # --- Research Stage ---
    google_trends: List[Dict]     # Raw trend data fetched from Google Trends.
    relevant_trends: List[Dict]   # Trend data filtered to be relevant to the request.
    research_findings: str        # The final research findings, either a trend summary or a Wikipedia abstract.

    # --- Writing & Revision Loop ---
    draft_post: str               # The generated draft article.
    critiques: Optional[List[str]]# A list of critiques for the draft.
    user_feedback: Optional[str]  # Feedback from human approval ('y' or 'n').
    fail_count: int               # A counter for auto-revision failures.


# ==============================================================================
# 4. Node Functions
# ==============================================================================
# Each function represents a working node in the graph, responsible for a specific task.

def preprocess_node(state: AgentState) -> dict:
    """
    Node 1: Preprocesses the user's request.
    Uses an LLM to convert the user's natural language request into structured data for subsequent nodes.
    - Reads: state['request']
    - Writes: 'industry', 'language', 'country_code', 'keywords'
    """
    print("--- Node: Preprocessing Input ---")
    request_text = state["request"]
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    structured_llm = llm.with_structured_output(RequestPreprocessModel)

    result = structured_llm.invoke(request_text)
    country_code = result.country_code or LANGUAGE_TO_COUNTRY.get(result.language, "US")

    print(f"    Extracted info: Industry='{result.industry}', Language='{result.language}', Country='{country_code}'")
    return {
        "industry": result.industry,
        "language": result.language,
        "country_code": country_code,
        "keywords": result.keywords
    }


def fetch_google_trends_node(state: AgentState) -> dict:
    """
    Node 2: Fetches Google Trends data.
    Retrieves current trending topics from Google Trends' RSS feed based on the country code from preprocessing.
    - Reads: state['country_code']
    - Writes: 'google_trends'
    """
    print("--- Node: Fetching Google Trends ---")
    country_code = state["country_code"]
    rss_url = f"https://trends.google.com/trending/rss?geo={country_code}"
    print(f"    Fetching data from {rss_url}...")
    feed = feedparser.parse(rss_url)

    if feed.bozo or not feed.entries:
        print("    Error: Failed to fetch or parse RSS feed.")
        return {"google_trends": []}

    # Extract and format the top 50 trends
    trends_data = []
    for entry in feed.entries[:50]:
        related_news = [{"title": entry.get('ht_news_item_title'), "url": entry.get('ht_news_item_url'),
                         "source": entry.get('ht_news_item_source')}] if entry.get('ht_news_item_title') else []
        trends_data.append({"trend_title": entry.title, "approximate_traffic": entry.get('ht_approx_traffic', 'N/A'),
                            "publication_date": entry.published, "related_news": related_news})

    print(f"    Successfully fetched and processed {len(trends_data)} trending topics.")
    return {"google_trends": trends_data}


def find_relevant_trends_node(state: AgentState) -> dict:
    """
    Node 3: Filters for relevant trends.
    Uses an LLM to analyze all fetched trends and identify the most relevant topics based on the user's request and industry context.
    - Reads: state['request'], state['industry'], state['google_trends']
    - Writes: 'relevant_trends'
    """
    print("--- Node: Filtering for Relevant Trends ---")
    user_request = state["request"]
    industry = state["industry"]
    all_trends = state["google_trends"]

    if not all_trends:
        print("    No trend data available for analysis.")
        return {"relevant_trends": []}

    # 1. Prepare data for LLM analysis: Combine each trend and its related news titles into a clear text block.
    trends_for_analysis = []
    for trend in all_trends:
        trend_title = trend.get("trend_title", "")
        related_news_titles = [news.get("title", "") for news in trend.get("related_news", []) if news.get("title")]
        related_news_str = ", ".join(related_news_titles) if related_news_titles else "None"
        analysis_text = f"Trend Title: {trend_title}\nRelated News: {related_news_str}"
        trends_for_analysis.append(analysis_text)
    trends_data_str = "\n\n".join(trends_for_analysis)

    # 2. Configure LLM and Prompt
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    structured_llm = llm.with_structured_output(RelevantTrends)
    system_prompt = (
        "You are a professional analysis assistant. Your task is to analyze the 'User Context' "
        "(containing the user request and industry), and read the 'Trending Topics Data' list provided below "
        "(each data point contains a 'Trend Title' and its associated 'Related News'). "
        "Please determine which trending topics are relevant to the 'User Context'. "
        "If a topic is relevant, you must return only its corresponding 'Trend Title'. "
        "Ensure the returned titles come exactly from the original list. "
        "If no topics are relevant, return an empty list."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User Context: '{context}'\n\nTrending Topics Data for Analysis:\n{trends_data}")
        ]
    )
    chain = prompt | structured_llm

    # 3. Invoke the LLM for analysis
    combined_context = f"User's Original Request: {user_request}\nSpecified Industry: {industry}"
    ai_response = chain.invoke({"context": combined_context, "trends_data": trends_data_str})
    relevant_titles = ai_response.trend_titles
    print(f"    AI identified relevant trend titles: {relevant_titles}")

    # 4. Filter the original data based on the titles returned by the LLM
    relevant_trends_details = [trend for trend in all_trends if trend["trend_title"] in relevant_titles]
    print(f"    Finalized {len(relevant_trends_details)} relevant trends.")
    return {"relevant_trends": relevant_trends_details}


def research_node(state: AgentState) -> dict:
    """
    Node 4a (Success Path): Compiles research findings.
    If relevant Google Trends were found, format them into a JSON string to serve as the basis for copywriting.
    - Reads: state['relevant_trends']
    - Writes: 'research_findings'
    """
    print("--- Node: Compiling Trend Research Findings ---")
    relevant_trends = state["relevant_trends"]
    findings_str = json.dumps(relevant_trends, indent=2, ensure_ascii=False)
    print(f"    Formatted {len(relevant_trends)} relevant trends into research material.")
    return {"research_findings": findings_str}


def fallback_wiki_search_node(state: AgentState) -> dict:
    """
    Node 4b (Fallback Path): Executes a Wikipedia search.
    When no relevant trends are found, uses the keywords from preprocessing to search Wikipedia and summarize the results.
    - Reads: state['keywords']
    - Writes: 'research_findings'
    """
    print("--- Node: No trends found, falling back to Wikipedia search ---")
    query = " ".join(state["keywords"])
    print(f"    Executing Wikipedia search for keywords: '{query}'...")
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=10000)
    tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    wiki_content = tool.run(query)

    if "No good Wikipedia Search Result was found" in wiki_content or not wiki_content:
        print("    No relevant content found on Wikipedia.")
        return {"research_findings": f"No research found on Wikipedia for the topic: '{query}'."}

    print(f"    Got content from Wikipedia, now summarizing...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert researcher. Summarize the following text into a concise, informative paragraph suitable for a content writer."),
        ("human", "Please summarize this document about '{topic}':\n\n{document}")])
    chain = prompt | llm
    summary = chain.invoke({"topic": query, "document": wiki_content}).content
    print("    Summary complete.")
    return {"research_findings": summary}


def copywriting_node(state: AgentState) -> dict:
    """
    Node 5: Writes the copy.
    This node writes an article based on the research findings. It has two modes:
    1. Initial Draft: Generates the first version of the copy based on 'research_findings'.
    2. Revision: If 'critiques' exist, revises the existing draft based on the feedback.
    It also ensures the output copy is in the specified language.
    - Reads: 'research_findings', 'industry', 'language', 'critiques' (optional), 'draft_post' (optional)
    - Writes: 'draft_post', 'critiques' (resets to None)
    """
    print("--- Node: Writing Copy ---")
    research_findings_str = state["research_findings"]
    critiques = state.get("critiques")
    language = state.get("language", "en")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

    if critiques:
        print(f"    Mode: Revising draft based on critiques (Language: {language})")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert writer. Revise the following draft based on the provided critiques. Ensure the final article aligns with the original research findings. IMPORTANT: The final article MUST be written in the following language: {language}."),
            ("human", "Original Research:\n{document}\n\nDraft to Revise:\n{draft}\n\nCritiques:\n{critiques}\n\nRevised Article:")])
        chain = prompt | llm
        response = chain.invoke({
            "language": language,
            "document": research_findings_str,
            "draft": state["draft_post"],
            "critiques": json.dumps(critiques, ensure_ascii=False)
        })
        return {"draft_post": response.content, "critiques": None}
    else:
        print(f"    Mode: Writing initial draft from research (Language: {language})")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a top-tier content writer. Your goal is to write a compelling, informative, and engaging article for the target industry based on the provided research document. IMPORTANT: The final article MUST be written in the following language: {language}."),
            ("human", "Target Industry: {industry}\n\nWrite a post based on this document:\n{document}")])
        chain = prompt | llm
        response = chain.invoke({
            "industry": state["industry"],
            "language": language,
            "document": research_findings_str
        })
        return {"draft_post": response.content}


def critic_node(state: AgentState) -> dict:
    """
    Node 6: Critiques the draft.
    Uses an LLM to evaluate the generated copy and provide revision suggestions in a structured format.
    Uses `with_structured_output` to ensure a robust JSON output.
    - Reads: 'request', 'research_findings', 'draft_post'
    - Writes: 'critiques'
    """
    print("--- Node: Critiquing Draft ---")
    critique_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that critiques draft articles."),
        ("user", "Given the original request:\n{request}\n\nAnd the research findings:\n{research_findings}\n\nPlease critique the following draft post:\n{draft_post}\n\nReturn a JSON list of critiques. If no critiques, return an empty list.")
    ])

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    structured_llm = llm.with_structured_output(Critiques)
    chain = critique_prompt | structured_llm

    try:
        response = chain.invoke(state)
        critiques = response.critique_list
    except Exception as e:
        print(f"    Could not generate structured critiques: {e}. Returning an empty list.")
        critiques = []

    print(f"    Generated critiques: {critiques}")
    return {"critiques": critiques}


def human_approval_node(state: AgentState):
    """
    Node 7: Human approval.
    Uses `interrupt` to pause the graph's execution, presenting the draft to a user for a decision.
    - Reads: 'draft_post'
    - Writes: 'user_feedback' (via the graph's resume mechanism)
    """
    print("--- Node: Awaiting Human Approval ---")
    # `interrupt` pauses execution and returns its contained data to the caller
    decision = interrupt({
        "message": "Do you approve this draft?",
        "draft_post": state["draft_post"]
    })
    user_feedback = decision.lower() if isinstance(decision, str) else "n"

    # 返回更新的状态字典，必须是 dict 类型
    return {
        "user_feedback": user_feedback
    }


# ==============================================================================
# 5. Conditional Edges
# ==============================================================================
# These functions determine the execution path of the graph.

def should_fallback_to_wiki(state: AgentState) -> str:
    """
    Decision 1: Determines the research path.
    Checks if relevant Google Trends were found.
    - If yes, proceed to the "research" path.
    - If no, proceed to the "fallback_to_wiki" path.
    """
    print("--- Decision: Choosing Research Path ---")
    if state.get("relevant_trends"):
        print("    Outcome: Relevant trends found. Proceeding to compile research.")
        return "research"
    else:
        print("    Outcome: No relevant trends found. Falling back to Wikipedia search.")
        return "fallback_to_wiki"


def after_critic_decide_next_step(state: AgentState) -> str:
    """
    Decision 2: Path selection after critique.
    Checks the 'critiques' from the review.
    - If there are critiques, it returns "redraft" for revision. It also checks failure counts to prevent infinite loops.
    - If there are no critiques, the draft is considered approved and proceeds to "request_human_approval".
    """
    print("--- Decision: Path After Critique ---")
    fail_count = state.get("fail_count", 0) + 1
    state["fail_count"] = fail_count

    if state.get("critiques"):
        print(f"    Outcome: Draft needs revision (Attempt #{fail_count}).")
        if fail_count >= MAX_FAILS:
            print(f"    Warning: Max revision attempts ({MAX_FAILS}) reached. Forcing a fallback to new research.")
            state["fail_count"] = 0  # Reset counter
            return "fallback_to_wiki"
        if fail_count >= MAX_TOTAL:
            print("    Error: Total rewrite limit reached. Terminating process.")
            return "end"
        return "redraft"
    else:
        state["fail_count"] = 0  # Reset counter on success
        print("    Outcome: Draft passed critique. Proceeding to human approval.")
        return "request_human_approval"


def after_human_feedback_decide_next_step(state: AgentState) -> str:
    """
    Decision 3: Path selection after human feedback.
    Determines the next step based on the user's input.
    - If the user approves ('y'), the process ends ("end").
    - If the user rejects, it returns to "redraft" and injects a generic rewrite instruction.
    """
    print("--- Decision: Path After Human Feedback ---")
    if state.get("user_feedback", "").lower() == 'y':
        print("    Outcome: Human approved. Ending process.")
        return "end"
    else:
        print("    Outcome: Human rejected. Returning to redraft.")
        state['critiques'] = ["User rejected the draft. Please rewrite from a different angle based on the original research."]
        return "redraft"


# ==============================================================================
# 6. Graph Construction
# ==============================================================================

# Initialize a memory saver for checkpointing to save and restore session state
memory = MemorySaver()

# Create the StateGraph instance and bind it to our AgentState
workflow = StateGraph(AgentState)

# --- Register all nodes ---
workflow.add_node("preprocess", preprocess_node)
workflow.add_node("fetch_trends", fetch_google_trends_node)
workflow.add_node("find_relevant", find_relevant_trends_node)
workflow.add_node("research", research_node)
workflow.add_node("fallback_wiki", fallback_wiki_search_node)
workflow.add_node("copywriting", copywriting_node)
workflow.add_node("critic", critic_node)
workflow.add_node("human_approval", human_approval_node)

# --- Define the graph's topology ---
# 1. Set the entry point
workflow.set_entry_point("preprocess")

# 2. Define regular edges
workflow.add_edge("preprocess", "fetch_trends")
workflow.add_edge("fetch_trends", "find_relevant")
workflow.add_edge("research", "copywriting")          # Success path converges on the copywriting node
workflow.add_edge("fallback_wiki", "copywriting")    # Fallback path also converges on the copywriting node
workflow.add_edge("copywriting", "critic")

# 3. Define conditional edges
workflow.add_conditional_edges(
    "find_relevant",
    should_fallback_to_wiki,
    {"research": "research", "fallback_to_wiki": "fallback_wiki"}
)
workflow.add_conditional_edges(
    "critic",
    after_critic_decide_next_step,
    {"redraft": "copywriting", "request_human_approval": "human_approval", "fallback_to_wiki": "fallback_wiki", "end": END}
)
workflow.add_conditional_edges(
    "human_approval",
    after_human_feedback_decide_next_step,
    {"redraft": "copywriting", "end": END}
)

# Compile the graph into a runnable application
app = workflow.compile(checkpointer=memory)
try:
    png_data = app.get_graph().draw_mermaid_png()
    with open("my_graph.png", "wb") as f:
        f.write(png_data)
    print("Graph saved to my_graph.png")
except Exception:
    pass


# ==============================================================================
# 7. Execution
# ==============================================================================

if __name__ == "__main__":
    # Create a unique thread ID for each independent run for state management
    thread_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": thread_id}}

    # --- Modify the user request here to test different paths ---
    # Test the standard path (Google Trends)
    request = "I want to know the latest news in the US sports world, write a report in English."
    # Test the fallback path (Wikipedia)
    # request = "Write an introduction to the history of the Roman Empire in German"

    print(f"--- Starting run with request: '{request}' ---\n")

    # First invocation of the graph, runs until the first interruption or end
    result = app.invoke({"request": request, "fail_count": 0}, config=thread_config)

    # Loop to handle interruptions until the graph completes
    while "__interrupt__" in result:
        print("\n✅ Graph paused, awaiting human approval.")
        print("===================================")
        print("Generated Draft:\n")
        print(result["__interrupt__"][0].value["draft_post"])
        print("===================================")

        # Collect human input
        human_input = input("Do you approve this draft? [y/n]: ")

        # Resume graph execution with human input using Command(resume=...)
        result = app.invoke(
            Command(resume=human_input),
            config=thread_config
        )

    # Process has ended normally, print the final state
    print("\n✅ Process finished. Final state:")
    # Print the final generated article
    if result.get("draft_post"):
      print("\n--- Final Article ---")
      print(result.get("draft_post"))

