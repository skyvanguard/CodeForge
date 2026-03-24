from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent
from utils.common import extract_jsons


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
- Write ONLY the Python code that solves the problem
- Include all necessary imports at the top
- The function/class must match the exact signature provided
- Do NOT include test code, examples, or explanations
- Do NOT wrap the code in markdown code blocks

Respond in JSON format:
<json>
{{
    "response": "... your complete Python code here ..."
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
