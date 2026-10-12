import re
from functools import lru_cache
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlglot import exp, parse
from sqlglot.errors import ParseError

from database import execute_query, get_schema
from llm import call_groq


MAX_RETRIES = 1
MAX_QUESTION_LENGTH = 500


# ---------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------


class AgentState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    """

    # User input
    question: str

    # Intent guard
    request_allowed: bool
    rejection_message: str | None

    # Database context
    schema: str

    # SQL generation
    sql: str
    safe_sql: str

    # Validation
    is_valid: bool
    validation_error: str | None

    # Execution
    result: Any
    execution_error: str | None

    # Agent control
    retry_count: int
    success: bool
    fatal_error: str | None

    # Observability
    attempts: list[dict[str, Any]]
    trace: list[str]


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------


def add_trace(
    state: AgentState,
    message: str,
) -> list[str]:
    """
    Add an event to the LangGraph execution trace.
    """

    trace = list(state.get("trace", []))
    trace.append(message)

    return trace


def clean_sql(model_output: str) -> str:
    """
    Remove Markdown and unnecessary prefixes
    from model-generated SQL.
    """

    cleaned = model_output.strip()

    # Remove opening ```sql or ``` block.
    cleaned = re.sub(
        r"^```(?:sql)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Remove closing ``` block.
    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    # Remove an optional SQL: prefix.
    cleaned = re.sub(
        r"^SQL\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    return cleaned.strip()


# ---------------------------------------------------------
# Groq SQL functions
# ---------------------------------------------------------


@lru_cache(maxsize=100)
def generate_sql(
    question: str,
    schema: str,
) -> str:
    """
    Convert a natural-language question into MySQL.

    Identical questions are cached while the application
    remains active.
    """

    system_prompt = (
        "Convert the question to one valid MySQL SELECT. "
        "Use only the supplied schema. "
        "Never modify data. "
        "Use explicit JOINs. "
        "Return SQL only, without Markdown."
    )

    user_prompt = f"""Schema:
{schema}

Question: {question}

SQL:"""

    response = call_groq(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=512,
    )

    return clean_sql(response)


def request_sql_repair(
    question: str,
    schema: str,
    failed_sql: str,
    error_message: str,
) -> str:
    """
    Ask Groq to repair invalid or failed SQL.
    """

    system_prompt = (
        "Repair the MySQL SELECT query using the schema "
        "and error. Return corrected SQL only."
    )

    user_prompt = f"""Schema:
{schema}

Question: {question}

Failed SQL:
{failed_sql}

Error:
{error_message}

Corrected SQL:"""

    response = call_groq(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=512,
    )

    return clean_sql(response)


# ---------------------------------------------------------
# Request-intent guard
# ---------------------------------------------------------


WRITE_REQUEST_PATTERNS = [
    # Direct destructive requests
    r"\b(delete|remove|erase|drop|truncate|purge)\b",

    # Requests to insert new university records
    (
        r"\b(insert|add|create|register|enroll)\b"
        r".*\b(student|course|department|enrollment|"
        r"record|row|data|marks?|grade|cgpa|attendance)\b"
    ),

    # Requests to modify existing university records
    (
        r"\b(update|modify|change|edit|replace|set)\b"
        r".*\b(student|course|department|enrollment|"
        r"record|row|data|marks?|grade|cgpa|attendance|"
        r"name|email)\b"
    ),

    # Raw SQL modification commands
    r"\binsert\s+into\b",
    r"\bupdate\s+\w+\s+set\b",
    r"\bdelete\s+from\b",
    r"\bcreate\s+table\b",
    r"\balter\s+table\b",
    r"\bdrop\s+table\b",
]


def detect_write_request(question: str) -> bool:
    """
    Detect whether the user is requesting a database
    insert, update, delete or schema modification.

    This runs before Groq is called.
    """

    normalized_question = " ".join(
        question.lower().split()
    )

    return any(
        re.search(
            pattern,
            normalized_question,
            flags=re.IGNORECASE,
        )
        for pattern in WRITE_REQUEST_PATTERNS
    )


# ---------------------------------------------------------
# SQL validation
# ---------------------------------------------------------


def validate_generated_sql(
    sql: str,
) -> tuple[bool, str | None, str | None]:
    """
    Validate model-generated SQL.

    Requirements:
    - Exactly one statement
    - Read-only query
    - No data or schema modification
    - Maximum 200 returned rows
    """

    if not sql or not sql.strip():
        return (
            False,
            "The model returned an empty query.",
            None,
        )

    try:
        statements = parse(
            sql,
            dialect="mysql",
        )

    except ParseError as error:
        return (
            False,
            f"Invalid SQL syntax: {error}",
            None,
        )

    if len(statements) != 1:
        return (
            False,
            "Only one SQL statement is allowed.",
            None,
        )

    statement = statements[0]

    prohibited_types = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Alter,
        exp.Create,
        exp.Command,
    )

    for prohibited_type in prohibited_types:
        if statement.find(prohibited_type):
            return (
                False,
                "Only read-only SELECT queries are allowed.",
                None,
            )

    if not isinstance(statement, exp.Query):
        return (
            False,
            "The generated statement is not a SELECT query.",
            None,
        )

    # Add a limit if one is not already present.
    if statement.find(exp.Limit) is None:
        statement = statement.limit(200)

    safe_sql = statement.sql(
        dialect="mysql",
        pretty=True,
    )

    return True, None, safe_sql


# ---------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------


def check_intent_node(
    state: AgentState,
) -> AgentState:
    """
    Node 1: Determine whether the user is requesting
    read-only analytics or a database modification.
    """

    question = state["question"]

    if detect_write_request(question):
        rejection_message = (
            "This analytics platform is read-only. "
            "I cannot insert, update, delete or modify "
            "university records. You can ask me to view, "
            "filter, compare, count or analyse the data."
        )

        return {
            "request_allowed": False,
            "rejection_message": rejection_message,
            "success": False,
            "trace": add_trace(
                state,
                "Rejected a database-modification request.",
            ),
        }

    return {
        "request_allowed": True,
        "rejection_message": None,
        "trace": add_trace(
            state,
            "Request classified as read-only analytics.",
        ),
    }


def reject_request_node(
    state: AgentState,
) -> AgentState:
    """
    Node 2: Terminate a database-modification request
    before calling Groq or MySQL.
    """

    rejection_message = state.get(
        "rejection_message",
        (
            "This platform supports only "
            "read-only analytics requests."
        ),
    )

    return {
        "success": False,
        "fatal_error": rejection_message,
        "trace": add_trace(
            state,
            "Agent stopped before SQL generation.",
        ),
    }


def load_schema_node(
    state: AgentState,
) -> AgentState:
    """
    Node 3: Retrieve the compact MySQL database schema.
    """

    try:
        schema = get_schema()

        return {
            "schema": schema,
            "fatal_error": None,
            "trace": add_trace(
                state,
                "Loaded the MySQL database schema.",
            ),
        }

    except Exception as error:
        return {
            "fatal_error": (
                "Could not retrieve the database schema: "
                f"{error}"
            ),
            "success": False,
            "trace": add_trace(
                state,
                "Database schema retrieval failed.",
            ),
        }


def generate_sql_node(
    state: AgentState,
) -> AgentState:
    """
    Node 4: Generate SQL using Groq.
    """

    try:
        sql = generate_sql(
            question=state["question"],
            schema=state["schema"],
        )

        return {
            "sql": sql,
            "safe_sql": "",
            "is_valid": False,
            "validation_error": None,
            "execution_error": None,
            "fatal_error": None,
            "trace": add_trace(
                state,
                "Generated SQL using the Groq model.",
            ),
        }

    except Exception as error:
        return {
            "fatal_error": (
                "The Groq SQL-generation request failed: "
                f"{error}"
            ),
            "success": False,
            "trace": add_trace(
                state,
                "SQL generation failed.",
            ),
        }


def validate_sql_node(
    state: AgentState,
) -> AgentState:
    """
    Node 5: Validate SQL using SQLGlot.
    """

    sql = state.get("sql", "")

    is_valid, error, safe_sql = validate_generated_sql(sql)

    attempts = list(state.get("attempts", []))

    attempts.append(
        {
            "attempt": state.get(
                "retry_count",
                0,
            ) + 1,
            "sql": safe_sql if is_valid else sql,
            "validation_error": error,
        }
    )

    if is_valid:
        trace_message = (
            "SQL passed the read-only safety validation."
        )
    else:
        trace_message = (
            "SQL validation failed and requires repair."
        )

    return {
        "is_valid": is_valid,
        "validation_error": error,
        "safe_sql": safe_sql or "",
        "attempts": attempts,
        "trace": add_trace(
            state,
            trace_message,
        ),
    }


def execute_sql_node(
    state: AgentState,
) -> AgentState:
    """
    Node 6: Execute validated SQL against MySQL.
    """

    safe_sql = state["safe_sql"]
    attempts = list(state.get("attempts", []))

    try:
        result = execute_query(safe_sql)

        if attempts:
            attempts[-1]["status"] = "success"

        return {
            "result": result,
            "execution_error": None,
            "success": True,
            "attempts": attempts,
            "trace": add_trace(
                state,
                "Executed the SQL query successfully.",
            ),
        }

    except Exception as error:
        error_message = str(error)

        if attempts:
            attempts[-1]["execution_error"] = (
                error_message
            )

        return {
            "execution_error": error_message,
            "success": False,
            "attempts": attempts,
            "trace": add_trace(
                state,
                "MySQL execution failed and requires repair.",
            ),
        }


def repair_sql_node(
    state: AgentState,
) -> AgentState:
    """
    Node 7: Repair SQL using validation or MySQL errors.
    """

    error_message = (
        state.get("validation_error")
        or state.get("execution_error")
        or "Unknown SQL error"
    )

    failed_sql = (
        state.get("safe_sql")
        or state.get("sql")
        or ""
    )

    try:
        repaired_sql = request_sql_repair(
            question=state["question"],
            schema=state["schema"],
            failed_sql=failed_sql,
            error_message=error_message,
        )

        return {
            "sql": repaired_sql,
            "safe_sql": "",
            "is_valid": False,
            "validation_error": None,
            "execution_error": None,
            "retry_count": state.get(
                "retry_count",
                0,
            ) + 1,
            "fatal_error": None,
            "trace": add_trace(
                state,
                (
                    "Repaired SQL using validation "
                    "or MySQL feedback."
                ),
            ),
        }

    except Exception as error:
        return {
            "fatal_error": (
                "The SQL-repair request failed: "
                f"{error}"
            ),
            "success": False,
            "trace": add_trace(
                state,
                "SQL repair failed.",
            ),
        }


def success_node(
    state: AgentState,
) -> AgentState:
    """
    Node 8: Mark the graph execution as successful.
    """

    return {
        "success": True,
        "fatal_error": None,
        "trace": add_trace(
            state,
            "Agent completed the analytics request.",
        ),
    }


def failure_node(
    state: AgentState,
) -> AgentState:
    """
    Node 9: Stop the graph with a failure.
    """

    final_error = (
        state.get("fatal_error")
        or state.get("validation_error")
        or state.get("execution_error")
        or "The agent could not complete the request."
    )

    return {
        "success": False,
        "fatal_error": final_error,
        "trace": add_trace(
            state,
            "Agent stopped without a successful result.",
        ),
    }


# ---------------------------------------------------------
# LangGraph routing functions
# ---------------------------------------------------------


def route_after_intent_check(
    state: AgentState,
) -> Literal["load_schema", "reject_request"]:
    """
    Route read-only questions to schema retrieval.
    Route write requests to rejection.
    """

    if state.get("request_allowed"):
        return "load_schema"

    return "reject_request"


def route_after_schema(
    state: AgentState,
) -> Literal["generate_sql", "failure"]:
    """
    Continue to SQL generation if the schema loaded.
    """

    if state.get("fatal_error"):
        return "failure"

    return "generate_sql"


def route_after_generation(
    state: AgentState,
) -> Literal["validate_sql", "failure"]:
    """
    Continue to validation if SQL generation succeeded.
    """

    if state.get("fatal_error"):
        return "failure"

    return "validate_sql"


def route_after_validation(
    state: AgentState,
) -> Literal[
    "execute_sql",
    "repair_sql",
    "failure",
]:
    """
    Execute valid SQL.

    Repair invalid SQL if a retry remains.
    Otherwise stop with failure.
    """

    if state.get("is_valid"):
        return "execute_sql"

    if state.get("retry_count", 0) < MAX_RETRIES:
        return "repair_sql"

    return "failure"


def route_after_execution(
    state: AgentState,
) -> Literal[
    "success",
    "repair_sql",
    "failure",
]:
    """
    Finish after successful execution.

    Repair failed execution if a retry remains.
    """

    if state.get("success"):
        return "success"

    if state.get("retry_count", 0) < MAX_RETRIES:
        return "repair_sql"

    return "failure"


def route_after_repair(
    state: AgentState,
) -> Literal["validate_sql", "failure"]:
    """
    Validate the repaired SQL or stop if repair failed.
    """

    if state.get("fatal_error"):
        return "failure"

    return "validate_sql"


# ---------------------------------------------------------
# Build and compile LangGraph
# ---------------------------------------------------------


def build_graph():
    """
    Build the complete Agentic NL2SQL workflow.
    """

    graph = StateGraph(AgentState)

    # Register graph nodes.
    graph.add_node(
        "check_intent",
        check_intent_node,
    )

    graph.add_node(
        "reject_request",
        reject_request_node,
    )

    graph.add_node(
        "load_schema",
        load_schema_node,
    )

    graph.add_node(
        "generate_sql",
        generate_sql_node,
    )

    graph.add_node(
        "validate_sql",
        validate_sql_node,
    )

    graph.add_node(
        "execute_sql",
        execute_sql_node,
    )

    graph.add_node(
        "repair_sql",
        repair_sql_node,
    )

    graph.add_node(
        "success",
        success_node,
    )

    graph.add_node(
        "failure",
        failure_node,
    )

    # Start with request-intent classification.
    graph.add_edge(
        START,
        "check_intent",
    )

    graph.add_conditional_edges(
        "check_intent",
        route_after_intent_check,
        {
            "load_schema": "load_schema",
            "reject_request": "reject_request",
        },
    )

    # Rejected requests terminate immediately.
    graph.add_edge(
        "reject_request",
        END,
    )

    # Continue after loading the schema.
    graph.add_conditional_edges(
        "load_schema",
        route_after_schema,
        {
            "generate_sql": "generate_sql",
            "failure": "failure",
        },
    )

    # Continue after SQL generation.
    graph.add_conditional_edges(
        "generate_sql",
        route_after_generation,
        {
            "validate_sql": "validate_sql",
            "failure": "failure",
        },
    )

    # Route after validation.
    graph.add_conditional_edges(
        "validate_sql",
        route_after_validation,
        {
            "execute_sql": "execute_sql",
            "repair_sql": "repair_sql",
            "failure": "failure",
        },
    )

    # Route after execution.
    graph.add_conditional_edges(
        "execute_sql",
        route_after_execution,
        {
            "success": "success",
            "repair_sql": "repair_sql",
            "failure": "failure",
        },
    )

    # Repaired SQL must be validated again.
    graph.add_conditional_edges(
        "repair_sql",
        route_after_repair,
        {
            "validate_sql": "validate_sql",
            "failure": "failure",
        },
    )

    # Terminal nodes.
    graph.add_edge(
        "success",
        END,
    )

    graph.add_edge(
        "failure",
        END,
    )

    return graph.compile()


sql_agent_graph = build_graph()


# ---------------------------------------------------------
# Public agent function used by Streamlit
# ---------------------------------------------------------


def run_agent(question: str) -> dict[str, Any]:
    """
    Validate the user input and run the LangGraph agent.
    """

    question = question.strip()

    if not question:
        return {
            "success": False,
            "sql": "",
            "error": "Question cannot be empty.",
            "attempts": [],
            "trace": [],
        }

    if len(question) > MAX_QUESTION_LENGTH:
        return {
            "success": False,
            "sql": "",
            "error": (
                "The question is too long. "
                f"Please keep it below "
                f"{MAX_QUESTION_LENGTH} characters."
            ),
            "attempts": [],
            "trace": [],
        }

    initial_state: AgentState = {
        "question": question,

        "request_allowed": False,
        "rejection_message": None,

        "retry_count": 0,
        "success": False,

        "fatal_error": None,
        "validation_error": None,
        "execution_error": None,

        "attempts": [],
        "trace": [],
    }

    try:
        final_state = sql_agent_graph.invoke(
            initial_state
        )

    except Exception as error:
        return {
            "success": False,
            "sql": "",
            "error": (
                "The LangGraph workflow failed: "
                f"{error}"
            ),
            "attempts": [],
            "trace": [
                (
                    "LangGraph encountered an "
                    "unexpected error."
                )
            ],
        }

    if final_state.get("success"):
        return {
            "success": True,
            "question": question,
            "sql": final_state.get(
                "safe_sql",
                final_state.get("sql", ""),
            ),
            "result": final_state["result"],
            "attempts": final_state.get(
                "attempts",
                [],
            ),
            "trace": final_state.get(
                "trace",
                [],
            ),
            "repaired": (
                final_state.get("retry_count", 0) > 0
            ),
        }

    return {
        "success": False,
        "question": question,
        "sql": final_state.get(
            "safe_sql",
            final_state.get("sql", ""),
        ),
        "error": final_state.get(
            "fatal_error",
            (
                "The agent could not answer "
                "the question."
            ),
        ),
        "attempts": final_state.get(
            "attempts",
            [],
        ),
        "trace": final_state.get(
            "trace",
            [],
        ),
    }