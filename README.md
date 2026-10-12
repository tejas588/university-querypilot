# University QueryPilot

University QueryPilot is a LangGraph-based Agentic NL2SQL analytics platform that converts natural-language university questions into safe MySQL queries.

## Features

- Natural-language-to-SQL generation
- LangGraph-based stateful workflow
- Input intent detection
- Schema-aware SQL generation
- SQLGlot AST validation
- Automatic SQL error repair
- Conditional routing and bounded retries
- SELECT-only MySQL access
- Streamlit analytics dashboard
- Plotly visualizations
- CSV result downloads
- Groq token-usage optimization

## Architecture

```text
User question
    ↓
Intent guard
    ↓
Schema retrieval
    ↓
Groq SQL generation
    ↓
SQLGlot validation
    ↓
MySQL execution
    ↓
Error-driven repair or successful result
    ↓
Table and visualization
```

## Tech Stack

- Python
- LangGraph
- Groq API
- GPT-OSS 20B
- MySQL
- SQLGlot
- Streamlit
- Pandas
- Plotly

## Project Structure

```text
university-querypilot/
├── app.py
├── database.py
├── llm.py
├── sql_agent.py
├── requirements.txt
├── database_setup.sql
├── .env.example
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/tejas588/university-querypilot.git
cd university-querypilot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create the database

Run `database_setup.sql` using MySQL Workbench.

Create a read-only application user:

```sql
CREATE USER IF NOT EXISTS 'university_reader'@'localhost'
IDENTIFIED BY 'your_password';

GRANT SELECT
ON university_db.*
TO 'university_reader'@'localhost';

FLUSH PRIVILEGES;
```

### 5. Configure environment variables

Copy `.env.example` to `.env` and provide your credentials:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=university_db
MYSQL_USER=university_reader
MYSQL_PASSWORD=your_mysql_password
```

Never commit `.env`.

### 6. Run the application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Example Questions

- Which department has the highest average CGPA?
- Show the top five students by CGPA.
- What is the average mark for each course?
- Which students have attendance below 85 percent?
- How many students belong to each department?

## Safety

The application uses defence in depth:

1. Input intent guard rejects modification requests.
2. SQLGlot validates generated SQL.
3. Only one read-only SQL statement is permitted.
4. Query results are limited.
5. MySQL access uses a SELECT-only account.
6. Failed SQL gets at most one repair attempt.

## Agent Workflow

The LangGraph workflow contains nodes for:

- Intent detection
- Schema retrieval
- SQL generation
- SQL validation
- Query execution
- Error-driven repair
- Success and failure termination

## Limitations

- Designed for a small university schema
- No conversational follow-up memory yet
- No user authentication
- Local development deployment
- LLM-generated SQL may still require semantic evaluation

## Future Improvements

- Conversational memory
- Human approval before SQL execution
- Automated NL2SQL evaluation dataset
- Persistent query history
- MCP server integration
- Role-based access control