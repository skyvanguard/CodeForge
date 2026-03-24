from agent.llm import DEFAULT_MODEL

QUESTION_ID = "problem_id"
GROUND_TRUTH_KEY = "expected_output"
MODEL = DEFAULT_MODEL


def format_input_dict(row):
    return {
        "domain": "coding",
        "problem_id": row["problem_id"],
        "description": row["description"],
        "function_signature": row["function_signature"],
        "difficulty": row.get("difficulty", "medium"),
    }
