from decimal import Decimal

import pandas as pd
import plotly.express as px
import streamlit as st

from sql_agent import run_agent


st.set_page_config(
    page_title="University QueryPilot",
    page_icon="🎓",
    layout="wide",
)


def convert_decimal_columns(dataframe):
    """
    Convert MySQL DECIMAL values into numeric values
    that Pandas and Plotly can recognize.
    """

    converted = dataframe.copy()

    for column in converted.columns:
        non_null_values = converted[column].dropna()

        if non_null_values.empty:
            continue

        first_value = non_null_values.iloc[0]

        if isinstance(first_value, Decimal):
            converted[column] = pd.to_numeric(
                converted[column],
                errors="coerce",
            )

    return converted


def format_name(name):
    """Convert database-style names into readable text."""

    return str(name).replace("_", " ").title()


def create_chart(dataframe):
    """
    Select and create a chart without another LLM call.
    """

    if dataframe.empty or len(dataframe) > 50:
        return None

    numeric_columns = list(
        dataframe.select_dtypes(
            include="number"
        ).columns
    )

    categorical_columns = [
        column
        for column in dataframe.columns
        if column not in numeric_columns
    ]

    if categorical_columns and numeric_columns:
        x_column = categorical_columns[0]
        y_column = numeric_columns[-1]

        return px.bar(
            dataframe,
            x=x_column,
            y=y_column,
            title=(
                f"{format_name(y_column)} by "
                f"{format_name(x_column)}"
            ),
            labels={
                x_column: format_name(x_column),
                y_column: format_name(y_column),
            },
            color=x_column,
        )

    if len(numeric_columns) >= 2:
        x_column = numeric_columns[0]
        y_column = numeric_columns[1]

        return px.scatter(
            dataframe,
            x=x_column,
            y=y_column,
            title=(
                f"{format_name(y_column)} versus "
                f"{format_name(x_column)}"
            ),
            labels={
                x_column: format_name(x_column),
                y_column: format_name(y_column),
            },
        )

    return None


def show_graph_trace(trace):
    """Display nodes visited by the LangGraph workflow."""

    if not trace:
        return

    st.subheader("LangGraph execution trace")

    for index, event in enumerate(trace, start=1):
        st.write(f"**{index}.** {event}")


def show_attempts(attempts):
    """Display SQL attempts and error feedback."""

    if not attempts:
        return

    with st.expander(
        f"SQL attempts ({len(attempts)})"
    ):
        for attempt in attempts:
            st.markdown(
                f"#### Attempt {attempt['attempt']}"
            )

            st.code(
                attempt.get("sql", ""),
                language="sql",
            )

            if attempt.get("validation_error"):
                st.error(
                    "Validation error: "
                    f"{attempt['validation_error']}"
                )

            if attempt.get("execution_error"):
                st.error(
                    "MySQL error: "
                    f"{attempt['execution_error']}"
                )

            if attempt.get("status") == "success":
                st.success(
                    "The query passed validation and executed."
                )


st.title("🎓 University QueryPilot")

st.caption(
    "A LangGraph-based Agentic NL2SQL analytics platform"
)


with st.sidebar:
    st.header("Example questions")

    example_questions = [
        "Which department has the highest average CGPA?",
        "Show the top five students by CGPA.",
        "What is the average mark for each course?",
        "Which students have attendance below 85 percent?",
        "How many students belong to each department?",
        "Show students who scored above 85 in any course.",
        "Which course has the highest average marks?",
    ]

    for example in example_questions:
        st.markdown(f"- {example}")

    st.divider()

    st.subheader("Agent workflow")

    st.markdown(
        """
        1. Load database schema  
        2. Generate SQL  
        3. Validate SQL  
        4. Execute SQL  
        5. Repair failed SQL  
        6. Return result
        """
    )

    st.subheader("Safety")

    st.info(
        "SQLGlot permits only read-only queries. "
        "The MySQL account also has SELECT-only access."
    )

    st.subheader("Groq usage")

    st.write(
        "Normally one Groq request is used. "
        "A second request is made only if SQL repair "
        "is required."
    )


question = st.text_input(
    "Ask a university-data question",
    placeholder=(
        "Example: Which department has the "
        "highest average CGPA?"
    ),
    max_chars=500,
)


submit_button = st.button(
    "Run LangGraph Agent",
    type="primary",
)


if submit_button:
    question = question.strip()

    if not question:
        st.warning("Please enter a question.")

    else:
        with st.spinner(
            "Running the LangGraph agent..."
        ):
            response = run_agent(question)

        if not response["success"]:
            error_message = response.get(
                "error",
                "The request could not be completed.",
            )

            if "read-only" in error_message.lower():
                st.warning(error_message)

            else:
                st.error(
                    "The agent could not answer the question."
                )
                st.write(error_message)

            st.write(
                response.get(
                    "error",
                    "An unknown error occurred.",
                )
            )

            failed_sql = response.get("sql")

            if failed_sql:
                st.subheader("Last generated SQL")

                st.code(
                    failed_sql,
                    language="sql",
                )

            show_graph_trace(
                response.get("trace", [])
            )

            show_attempts(
                response.get("attempts", [])
            )

        else:
            result = convert_decimal_columns(
                response["result"]
            )

            if response.get("repaired"):
                st.success(
                    "The LangGraph agent repaired the SQL "
                    "and completed the request."
                )
            else:
                st.success(
                    "The LangGraph agent completed "
                    "the request successfully."
                )

            sql_tab, result_tab, chart_tab, trace_tab = st.tabs(
                [
                    "Generated SQL",
                    "Query Result",
                    "Visualization",
                    "Agent Trace",
                ]
            )

            with sql_tab:
                st.code(
                    response["sql"],
                    language="sql",
                )

                st.caption(
                    "The query passed SQLGlot validation "
                    "before execution."
                )

            with result_tab:
                if result.empty:
                    st.info(
                        "The query executed successfully "
                        "but returned no records."
                    )

                else:
                    st.dataframe(
                        result,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.caption(
                        f"{len(result)} row(s) returned."
                    )

                    csv_data = result.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download result as CSV",
                        data=csv_data,
                        file_name="query_result.csv",
                        mime="text/csv",
                    )

            with chart_tab:
                if result.empty:
                    st.info(
                        "A chart cannot be created because "
                        "the query returned no data."
                    )

                else:
                    chart = create_chart(result)

                    if chart is None:
                        st.info(
                            "A useful chart could not be "
                            "created automatically for this result."
                        )

                    else:
                        st.plotly_chart(
                            chart,
                            use_container_width=True,
                        )

            with trace_tab:
                show_graph_trace(
                    response.get("trace", [])
                )

                show_attempts(
                    response.get("attempts", [])
                )