import json
import sys
from typing import Dict, Any, List

# Define expected formats
if __package__:
    from .qwen_normalization import EXPECTED_FORMATS as QWEN_FORMATS
else:
    from qwen_normalization import EXPECTED_FORMATS as QWEN_FORMATS

# Preserve the filter's historical subset.
EXPECTED_FORMATS = {
    key: value
    for key, value in QWEN_FORMATS.items()
    if key != "qwen_overlapping_speech"
}


def filter_json_object(json_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter a JSON object to keep only the 'key' and keys matching EXPECTED_FORMATS
    """
    if not isinstance(json_obj, dict):
        return {}

    filtered_obj = {}

    # Always keep the "key" key if it exists
    if "key" in json_obj:
        filtered_obj["key"] = json_obj["key"]

    # Only keep other keys if they match one in EXPECTED_FORMATS
    for key, value in json_obj.items():
        if key in EXPECTED_FORMATS:
            if EXPECTED_FORMATS[key]["type"] == "category":
                if value in EXPECTED_FORMATS[key]["categories"]:
                    filtered_obj[key] = value
            elif EXPECTED_FORMATS[key]["type"] == "number":
                try:
                    filtered_obj[key] = int(value)
                except:
                    continue

    return filtered_obj


def process_jsonl_file(input_file: str, output_file: str) -> None:
    """
    Process a JSONL file and create a filtered version
    """
    filtered_objects = []

    # Read and process each line
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                json_obj = json.loads(line.strip())
                filtered_obj = filter_json_object(json_obj)
                filtered_objects.append(filtered_obj)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line[:50]}...")

    # Write filtered objects to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        for obj in filtered_objects:
            f.write(json.dumps(obj) + "\n")

    print(f"Processed {len(filtered_objects)} objects")
    print(f"Filtered JSONL written to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_jsonl.py input.jsonl output.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_jsonl_file(input_file, output_file)
