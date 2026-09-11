#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Qwen2-Audio JSON Output Standardizer

This script standardizes the existing JSON outputs from Qwen2-Audio
by applying LLM-based or rules-based refinement.
"""

import json
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Union, Any
from tqdm import tqdm

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except ImportError:
    logging.warning(
        "Transformers not found. Please install with: pip install transformers"
    )
    AutoTokenizer, AutoModelForCausalLM, pipeline = None, None, None

if __package__:
    from .qwen_normalization import (
        EXPECTED_FORMATS,
        TRANSLATION_DICT,
        translate_non_english,
        process_llm_output,
        rules_based_standardize,
    )
else:
    from qwen_normalization import (
        EXPECTED_FORMATS,
        TRANSLATION_DICT,
        translate_non_english,
        process_llm_output,
        rules_based_standardize,
    )


class JsonOutputStandardizer:
    """
    A class for standardizing existing JSON outputs from Qwen2-Audio.
    """

    def _translate_non_english(self, text):
        """Translate known terms using this standardizer instance translation dictionary."""
        return translate_non_english(text, self.translation_dict)

    def _process_llm_output(self, metric_name, llm_output):
        """Clean an LLM response using this instance schema and the shared output rules."""
        return process_llm_output(metric_name, llm_output, self.expected_formats)

    def _rules_based_standardize(self, metric_name, raw_output):
        """Apply deterministic category and numeric rules using this instance schema."""
        return rules_based_standardize(
            metric_name, raw_output, self.expected_formats, self.translation_dict
        )

    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", device="auto"):
        """
        Initialize the standardizer with a text-only LLM.

        Args:
            model_name (str): HuggingFace model name for the text-only LLM
            device (str): Device to run the model on ("cpu", "cuda", "auto")
        """
        self.expected_formats = EXPECTED_FORMATS
        self.translation_dict = TRANSLATION_DICT
        self.llm_available = False

        # Try to initialize the LLM
        if AutoTokenizer is not None and AutoModelForCausalLM is not None:
            try:
                logging.info(f"Initializing language model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, low_cpu_mem_usage=True, torch_dtype="auto"
                )

                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=device,
                    max_new_tokens=512,
                )

                self.llm_available = True
                logging.info("LLM initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize LLM: {e}")
                self.llm_available = False

    def _generate_prompt(self, metric_name, raw_output):
        """
        Generate a prompt for the text-only LLM to standardize the output.

        Args:
            metric_name (str): Name of the metric being standardized
            raw_output (str): Raw output from Qwen2-Audio

        Returns:
            str: Prompt for the text-only LLM
        """
        if metric_name not in self.expected_formats:
            return f"""
            I need to classify this speech property: "{raw_output}"
            Please extract the most relevant category mentioned.
            Provide only the category name without explanations.
            """

        format_info = self.expected_formats[metric_name]

        if format_info["type"] == "number":
            return f"""
            Extract the number of speakers from this response: "{raw_output}"
            If multiple numbers are mentioned, choose the most definitive one.
            Provide only the number (1-10) without any other text.
            """

        elif format_info["type"] == "multi-category":
            categories = ", ".join([f'"{cat}"' for cat in format_info["categories"]])
            return f"""
            Classify this response into one or more of these categories: {categories}
            Response: "{raw_output}"
            Extract only the most relevant categories from the valid list.
            If 'Other' is chosen, specify what it is if possible (e.g., "Other: Polish").
            Provide only the category name(s) without explanations, separated by commas if multiple.
            """

        else:  # "category"
            categories = ", ".join([f'"{cat}"' for cat in format_info["categories"]])
            return f"""
            Classify this response into exactly one of these categories: {categories}
            Response: "{raw_output}"
            Find the closest match from the valid categories.
            Provide only the exact category name without explanations.
            """

    def standardize(self, metric_name, raw_output):
        """
        Standardize the output from Qwen2-Audio.

        Args:
            metric_name (str): Name of the metric
            raw_output (Union[str, int]): Raw output from Qwen2-Audio

        Returns:
            Union[str, int, List[str]]: Standardized output
        """
        # Convert int to str if needed
        if isinstance(raw_output, int):
            raw_output = str(raw_output)

        # First translate any non-English terms
        translated_output = self._translate_non_english(raw_output)

        # If LLM is available, use it
        if self.llm_available:
            try:
                prompt = self._generate_prompt(metric_name, translated_output)

                # Generate response from the text-only LLM
                response = self.pipeline(
                    prompt, do_sample=False, return_full_text=False
                )[0]["generated_text"]

                # Process the LLM's output
                return self._process_llm_output(metric_name, response)

            except Exception as e:
                logging.warning(
                    f"LLM processing failed for {metric_name}: {e}. Falling back to rules-based approach."
                )
                return self._rules_based_standardize(metric_name, translated_output)
        else:
            # Fall back to rules-based approach
            return self._rules_based_standardize(metric_name, translated_output)

    def standardize_json(self, qwen_output):
        """
        Standardize an entire JSON output from Qwen2-Audio.

        Args:
            qwen_output (dict): Dictionary containing Qwen2-Audio outputs

        Returns:
            dict: Dictionary with standardized outputs
        """
        # Create a copy to not modify the original
        standardized = qwen_output.copy()

        # Keep track of the original values
        original_values = {}

        # Process each metric in the output
        for key, value in qwen_output.items():
            # Skip non-qwen keys
            if not key.startswith("qwen_"):
                continue

            # Store original value
            original_values[key] = value

            # Standardize the value
            standardized[key] = self.standardize(key, value)

        # Add the original values for reference
        standardized["original_values"] = original_values

        return standardized


def process_json_file(file_path, output_path=None, use_llm=True):
    """
    Process a single JSON file containing Qwen2-Audio outputs.

    Args:
        file_path (str): Path to the JSON file
        output_path (str): Path to save the standardized JSON (default: original_path_standardized.json)
        use_llm (bool): Whether to use LLM-based standardization

    Returns:
        dict: Standardized output
    """
    # Set default output path if not provided
    if output_path is None:
        file_path_obj = Path(file_path)
        output_path = file_path_obj.with_stem(f"{file_path_obj.stem}_standardized")

    # Load the JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        qwen_output = json.load(f)

    # Initialize the standardizer
    standardizer = JsonOutputStandardizer()

    # Standardize the output
    standardized = standardizer.standardize_json(qwen_output)

    # Save the standardized output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(standardized, f, indent=2, ensure_ascii=False)

    return standardized


def process_directory(input_dir, output_dir=None, use_llm=True, file_pattern="*.json"):
    """
    Process all JSON files in a directory.

    Args:
        input_dir (str): Directory containing JSON files
        output_dir (str): Directory to save standardized JSON files
        use_llm (bool): Whether to use LLM-based standardization
        file_pattern (str): Glob pattern for JSON files

    Returns:
        dict: Summary of processing results
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(input_dir), f"{os.path.basename(input_dir)}_standardized"
        )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of JSON files
    input_path = Path(input_dir)
    json_files = list(input_path.glob(file_pattern))

    if not json_files:
        logging.warning(
            f"No JSON files found in {input_dir} matching pattern {file_pattern}"
        )
        return {"error": "No JSON files found"}

    logging.info(f"Found {len(json_files)} JSON files to process")

    # Process each file
    results = {}
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        try:
            # Generate output path
            rel_path = json_file.relative_to(input_path)
            output_file = Path(output_dir) / rel_path.with_stem(
                f"{json_file.stem}_standardized"
            )

            # Ensure output directory exists
            os.makedirs(output_file.parent, exist_ok=True)

            # Process the file
            standardized = process_json_file(json_file, output_file, use_llm)

            results[str(json_file)] = {"output_file": str(output_file), "success": True}

        except Exception as e:
            logging.error(f"Error processing {json_file}: {e}")
            results[str(json_file)] = {"success": False, "error": str(e)}

    # Create summary
    summary = {
        "total_files": len(json_files),
        "processed": sum(1 for r in results.values() if r.get("success", False)),
        "failed": sum(1 for r in results.values() if not r.get("success", False)),
        "use_llm": use_llm,
    }

    # Save summary
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "results": results}, f, indent=2, ensure_ascii=False
        )

    logging.info(f"Processing complete. Summary saved to {summary_path}")
    return summary


def main():
    """Parse CLI inputs, configure the standardizer, and write normalized output files."""
    parser = argparse.ArgumentParser(description="Standardize Qwen2-Audio JSON outputs")
    parser.add_argument("input", help="Input JSON file or directory")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file or directory (default: adds '_standardized' to input name)",
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="Use rules-based approach instead of LLM"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Setup logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    logging.basicConfig(
        level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    input_path = Path(args.input)

    # If input is a directory, process all JSON files
    if input_path.is_dir():
        summary = process_directory(input_path, args.output, use_llm=not args.no_llm)

        print("\nProcessing Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")

    # If input is a file, process single file
    elif input_path.is_file():
        try:
            standardized = process_json_file(
                input_path, args.output, use_llm=not args.no_llm
            )
            print(f"Successfully standardized {input_path}")

        except Exception as e:
            logging.error(f"Error processing {input_path}: {e}")
            print(f"Failed to standardize {input_path}: {e}")

    else:
        logging.error(f"Input path {input_path} does not exist")
        print(f"Error: Input path {input_path} does not exist")


if __name__ == "__main__":
    main()
