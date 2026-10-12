import os

import mysql.connector
import pandas as pd
from dotenv import load_dotenv
from functools import lru_cache


load_dotenv()


def get_connection():
    """Create and return a MySQL database connection."""

    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.getenv("MYSQL_DATABASE"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
    )


@lru_cache(maxsize=1)
def get_schema():
    """
    Return a compact database schema.

    The result is cached because the database structure
    does not change while the application is running.
    """

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            SELECT
                c.TABLE_NAME,
                c.COLUMN_NAME,
                c.COLUMN_TYPE,
                c.COLUMN_KEY,
                k.REFERENCED_TABLE_NAME,
                k.REFERENCED_COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS c
            LEFT JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE k
                ON c.TABLE_SCHEMA = k.TABLE_SCHEMA
                AND c.TABLE_NAME = k.TABLE_NAME
                AND c.COLUMN_NAME = k.COLUMN_NAME
            WHERE c.TABLE_SCHEMA = %s
            ORDER BY
                c.TABLE_NAME,
                c.ORDINAL_POSITION
            """,
            (os.getenv("MYSQL_DATABASE"),),
        )

        rows = cursor.fetchall()
        tables = {}

        for row in rows:
            table_name = row["TABLE_NAME"]
            column_name = row["COLUMN_NAME"]
            column_type = row["COLUMN_TYPE"]

            column_description = (
                f"{column_name} {column_type}"
            )

            if row["COLUMN_KEY"] == "PRI":
                column_description += " PK"

            if row["REFERENCED_TABLE_NAME"]:
                column_description += (
                    f" FK->{row['REFERENCED_TABLE_NAME']}."
                    f"{row['REFERENCED_COLUMN_NAME']}"
                )

            tables.setdefault(
                table_name,
                [],
            ).append(column_description)

        schema_lines = []

        for table_name, columns in tables.items():
            schema_lines.append(
                f"{table_name}({', '.join(columns)})"
            )

        return "\n".join(schema_lines)

    finally:
        cursor.close()
        connection.close()


def execute_query(sql):
    """Execute a SELECT query and return a DataFrame."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(sql)

        rows = cursor.fetchall()
        columns = [
            description[0]
            for description in cursor.description
        ]

        return pd.DataFrame(rows, columns=columns)

    finally:
        connection.close()

'''
if __name__ == "__main__":
    print(get_schema())

    result = execute_query(
        """
        SELECT
            s.student_name,
            d.department_code,
            s.cgpa
        FROM students s
        JOIN departments d
            ON s.department_id = d.department_id
        ORDER BY s.cgpa DESC
        """
    )

    print(result)
'''
