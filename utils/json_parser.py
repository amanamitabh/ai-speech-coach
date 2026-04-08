import json
import re

def parse_llm_output(output: str):
    try:
        # Remove markdown code block markers
        cleaned = re.sub(r"```(?:json)?", "", output)
        cleaned = cleaned.replace("```", "").strip()

        # Extract JSON
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")

        json_str = match.group(0)
        data = json.loads(json_str)

    except Exception as e:
        print("JSON parsing failed:", e)
        return {
            "cleaned_text": output.strip(),
            "fillers": {},
            "total_fillers": 0
        }

    # Return data as a dictionary
    return {
        "cleaned_text": data.get("cleaned_text", "").strip(),
        "fillers": data.get("fillers", {}),
        "total_fillers": data.get("total_fillers", 0)
    }