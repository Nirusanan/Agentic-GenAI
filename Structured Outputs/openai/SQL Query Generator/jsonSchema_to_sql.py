import os
import json
from structured_ouput_query import struct_output_generate
import gradio as gr

def json_to_sql(query_json):
    table_name = query_json.get("table_name")
    columns = query_json.get("columns", [])
    conditions = query_json.get("conditions", [])
    group_by = query_json.get("group_by", [])
    order_by = query_json.get("order_by", [])  # Assume it's now a list of dicts for more precise control
    limit = query_json.get("limit")  # Optional limit for top results

    # Process columns for SELECT clause
    select_columns = []
    for col in columns:
        if isinstance(col, dict):  # If column is an aggregate function
            function = col.get("function")
            column = col.get("column")
            alias = col.get("alias", "")
            if alias:
                select_columns.append(f"{function}({column}) AS {alias}")
            else:
                select_columns.append(f"{function}({column})")
        else:  # Regular column
            select_columns.append(col)

    select_clause = ", ".join(select_columns)

    # Process conditions for WHERE clause
    where_conditions = []
    for condition in conditions:
        column = condition.get("column")
        operator = condition.get("operator")
        value = condition.get("value")

        # If the condition value is a string, add quotes
        if isinstance(value, str):
            value = f"'{value}'"

        where_conditions.append(f"{column} {operator} {value}")

    where_clause = " AND ".join(where_conditions) if where_conditions else ""

    # Process group by for GROUP BY clause
    group_by_clause = ", ".join(group_by) if group_by else ""


    # Construct ORDER BY clause
    if order_by:
        order_by_clause = ", ".join(
            [f"{col.get('column')} {col.get('direction', 'asc')}" for col in order_by]
        )
        order_by_clause = f"ORDER BY {order_by_clause}"
    else:
        order_by_clause = ""

    # Add LIMIT clause if specified
    limit_clause = f"LIMIT {limit}" if limit is not None else ""

    # Build query with non-empty clauses
    query_parts = [
        f"SELECT {select_clause}",
        f"FROM {table_name}",
        f"WHERE {where_clause}" if where_clause else "",
        f"GROUP BY {group_by_clause}" if group_by else "",
        f"{order_by_clause}" if order_by_clause else "",
        f"{limit_clause}" if limit_clause else ""
    ]

    query = "\n".join(part for part in query_parts if part)

    return query



def query_generate(question):
    struct_out = struct_output_generate(question)
    struct_out_dict = json.loads(struct_out)
    sql = json_to_sql(struct_out_dict)
    return sql


demo = gr.Interface(
    fn=query_generate,
    inputs=gr.Textbox(label="Ask the question", placeholder="Who are the top 5 customers by total spending?"),
    outputs=gr.Textbox(label="SQL Query"),
    title="Sql Query Generator",
)

demo.launch()