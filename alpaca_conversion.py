"""Loads regen_compose.jsonl, and converts it to a format that can be used by Alpaca eval."""
import json

from pathlib import Path
from typing import List, Dict, Any


def load_json(path: Path) -> List[Dict[str, Any]]:
    """Loads json file from path."""
    with open(path, "r") as f:
        return json.load(f)


def format_json_line(line: dict) -> dict:
    """Changes json format into alpaca-suitable format.

    That is, goes from:
    ```
    {
      "instruction": "Convert the time \"14:00...
      "input": "",
      "output": "The time 14:00 in 24-hour tim...
    }
    ```
    to this
    ```
    {
      "instruction": "Convert the time \"14:00...
      "dataset": "sequential_instructions",
      "output": "The time 14:00 in 24-hour time...
    }
    ```
    """
    return {
        "instruction": line["instruction"],
        "dataset": "sequential_instructions",
        "output": line["output"],
    }


def main():
    """Converts regen_compose.jsonl to a format that can be used by Alpaca eval."""
    regen_compose_path = Path("regen_compose.json")
    assert regen_compose_path.exists(), f"{regen_compose_path} does not exist."

    regen_compose = load_json(regen_compose_path)
    print("Formatting Instructions...")
    formatted_regen_compose = [format_json_line(line) for line in regen_compose]
    print("Done.")

    print("Saving file...")
    with open("regen_compose_alpaca_format.json", "w") as f:
        json.dump(formatted_regen_compose, f)
    print("Done saving file.")


if __name__ == "__main__":
    main()
