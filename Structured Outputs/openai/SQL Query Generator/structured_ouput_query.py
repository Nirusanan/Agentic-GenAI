from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def struct_output_generate(question):
    completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant. The current date is Jan 02, 2025. You help users query for the data they are looking for by calling the query function."},
            {
                "role": "user",
                "content": question
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "query",
                "schema": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "enum": ["sales_orders"]
                        },
                        "columns": {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {
                                        "type": "string",
                                        "enum": [
                                            "id",
                                            "customer_id",
                                            "customer_name",
                                            "sales_person_id",
                                            "sales_person_name",
                                            "item_no",
                                            "item_name",
                                            "price",
                                            "quantity",
                                            "total_amount",
                                            "created_at"
                                        ]
                                    },
                                    {
                                        "type": "object",
                                        "properties": {
                                            "function": {
                                                "type": "string",
                                                "enum": ["SUM", "COUNT", "AVG", "DISTINCT"]
                                            },
                                            "column": {
                                                "type": "string",
                                                "enum": [
                                                    "*",
                                                    "id",
                                                    "customer_id",
                                                    "customer_name",
                                                    "sales_person_id",
                                                    "sales_person_name",
                                                    "item_no",
                                                    "item_name",
                                                    "price",
                                                    "quantity",
                                                    "total_amount",
                                                    "created_at"
                                                ]
                                            },
                                            "alias": {
                                                "type": "string"
                                            }
                                        },
                                        "required": ["function", "column", "alias"],
                                        "additionalProperties": False
                                    }
                                ]
                            }
                        },
                        "conditions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string"},
                                    "operator": {
                                        "type": "string",
                                        "enum": ["=", ">", "<", "!=", "LIKE", "IN", "NOT IN"]
                                    },
                                    "value": {
                                        "anyOf": [
                                            {"type": "string"},
                                            {"type": "number"},
                                            {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            {
                                                "type": "object",
                                                "properties": {
                                                    "column_name": {"type": "string"}
                                                },
                                                "required": ["column_name"],
                                                "additionalProperties": False
                                            }
                                        ]
                                    }
                                },
                                "required": ["column", "operator", "value"],
                                "additionalProperties": False
                            }
                        },
                        "group_by": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "customer_id",
                                    "customer_name",
                                    "sales_person_id",
                                    "sales_person_name",
                                    "item_no",
                                    "item_name",
                                    "created_at"
                                ]
                            }
                        },
                        
                        "order_by": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {
                                        "type": "string"
                                    },
                                    "direction": {
                                        "type": "string",
                                        "enum": ["asc", "desc"]
                                    }
                                },
                                "required": ["column", "direction"],
                                "additionalProperties": False
                            }
                        },
                        "limit": {
                            "type": "integer"
                        }
                    },
                    "required": ["table_name", "columns", "conditions", "group_by", "order_by",
                                 "limit"],
                    "additionalProperties": False
                },
                "strict": True
            }

        }

    )

    return completion.choices[0].message.content