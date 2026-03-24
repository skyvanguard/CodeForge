import re

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent
from utils.common import extract_jsons


def extract_code(text):
    """Extract Python code from LLM response. Tries JSON first, then markdown blocks, then raw text."""
    # Try JSON extraction first
    try:
        jsons = extract_jsons(text)
        if jsons and "response" in jsons[-1]:
            return jsons[-1]["response"]
    except Exception:
        pass

    # Try markdown code blocks
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    # Fallback: if text contains def/class, use it as-is
    if "def " in text or "class " in text:
        return text.strip()

    return None


class TaskAgent(AgentSystem):
    def forward(self, inputs):
        domain = inputs.get("domain", "")

        if domain == "coding":
            return self._solve_coding(inputs)
        return self._solve_generic(inputs)

    def _solve_coding(self, inputs):
        description = inputs.get("description", "")
        function_signature = inputs.get("function_signature", "")

        instruction = f"""You are an expert Python programmer. Solve the following coding problem.

## Problem
{description}

## Function Signature
```python
{function_signature}
```

## Requirements
- Write ONLY the Python function/class that solves the problem
- Include all necessary imports at the top
- The function/class must match the exact signature provided
- Do NOT include test code, examples, or explanations
- Respond with ONLY the code, nothing else"""

        new_msg_history = chat_with_agent(
            instruction, model=self.model, msg_history=[], logging=self.log
        )

        prediction = None
        try:
            response_text = new_msg_history[-1].get("text", "")
            prediction = extract_code(response_text)
        except Exception as e:
            self.log(f"Error extracting prediction: {e}")

        if prediction is None:
            prediction = "None"

        return prediction, new_msg_history

    def _solve_generic(self, inputs):
        instruction = f"""You are an agent.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": ...
}}
</json>"""
        new_msg_history = chat_with_agent(
            instruction, model=self.model, msg_history=[], logging=self.log
        )

        prediction = "None"
        try:
            extracted_jsons = extract_jsons(new_msg_history[-1]["text"])
            if extracted_jsons is not None and "response" in extracted_jsons[-1]:
                prediction = extracted_jsons[-1]["response"]
        except Exception as e:
            self.log(f"Error extracting prediction: {e}")
            prediction = "None"

        return prediction, new_msg_history
