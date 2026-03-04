"""
Run the full agent workflow (retrieval + generation + validation + MCP context) on the Glint SmartThings dataset.

Uses the MCP server to fetch live openHAB context (items, things, rules) when ENABLE_CONTEXT_VALIDATION
is true (default), so generated rules are validated against your actual system.

From repo root:

  python3 agent_pipeline/run_glint_pipeline.py                    # all 185 examples
  python3 agent_pipeline/run_glint_pipeline.py --id 0             # single example

Output:
  generated_rules/results/agents_workflow/glint_smartthings_rule/<id>.rules
  generated_rules/results/agents_workflow/glint_smartthings_rule/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

try:
    from main import run_generation_loop, maybe_deploy_via_mcp
except ModuleNotFoundError:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from main import run_generation_loop, maybe_deploy_via_mcp
from tools.context_loader import load_contexts
from tools.context_fetcher import SystemContextFetcher, SystemContext
from tools.rule_parser import RuleParser


DATASET_PATH = Path("datasets") / "glint_smartthings_rule.json"
RESULTS_DIR = (
    Path("generated_rules") / "results" / "agents_workflow" / "glint_smartthings_rule"
)


def fetch_system_context() -> SystemContext | None:
    """
    Fetch live system context via the MCP server (items, things, rules).

    Controlled by ENABLE_CONTEXT_VALIDATION. When true, the pipeline validates
    generated rules against your openHAB instance via MCP.
    """
    enable_context = os.environ.get("ENABLE_CONTEXT_VALIDATION", "true").lower() != "false"
    if not enable_context:
        print("Context validation disabled via ENABLE_CONTEXT_VALIDATION.")
        return None

    print("\n=== Fetching live system context via MCP server (Glint pipeline) ===")
    try:
        fetcher = SystemContextFetcher()
        context = fetcher.fetch_all()
        print("✓ System context loaded successfully\n")
        return context
    except Exception as exc:
        print(f"⚠ Warning: Could not fetch system context: {exc}")
        print("Proceeding without context-aware validation.\n")
        return None


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load the Glint dataset (id, text) pairs."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list at top level in {path}, got {type(data)}")
    return data


def run_agent_pipeline_on_glint_dataset(
    only_id: int | None = None,
    delay: float = 13.0,
    no_deploy: bool = False,
) -> None:
    """
    Run the full agent pipeline on the Glint SmartThings dataset.

    For each example (or a single one if only_id is set):
      - Retrieves docs from context/
      - Fetches live openHAB context via MCP (if ENABLE_CONTEXT_VALIDATION=true)
      - Runs generation + syntax + context validation loop
      - Writes <id>.rules and summary.json under RESULTS_DIR.

    Supports resuming: already-completed ids found in summary.json are skipped.
    """
    print(f"Loading Glint dataset from {DATASET_PATH} ...")
    examples = load_dataset(DATASET_PATH)

    load_dotenv()
    retriever = load_contexts(path="./context", vs_path="./vectorstore/faiss")
    system_context = fetch_system_context()

    max_attempts = int(os.environ.get("GENERATION_MAX_ATTEMPTS", "3"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary.json"

    # Load already-completed results so we can resume without re-calling the API
    results: List[Dict[str, Any]] = []
    completed_ids: set = set()
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as sf:
                existing = json.load(sf)
            for entry in existing:
                completed_ids.add(entry.get("id"))
                results.append(entry)
            if completed_ids:
                print(f"Resuming – skipping {len(completed_ids)} already-completed id(s).")
        except Exception:
            pass

    first_request = True
    for example in examples:
        ex_id = example.get("id")
        if only_id is not None and ex_id != only_id:
            continue
        if ex_id in completed_ids:
            continue

        text = (example.get("text") or "").strip()

        # Throttle to stay within the free-tier RPM limit
        if not first_request and delay > 0:
            print(f"  (waiting {delay}s to respect RPM limit...)")
            time.sleep(delay)
        first_request = False

        print(f"\n=== Agent pipeline (Glint) id={ex_id} ===")
        print(f"Request: {text}")

        generation, validation, is_valid, attempts_used = run_generation_loop(
            request=text,
            retriever=retriever,
            max_attempts=max_attempts,
            system_context=system_context,
        )

        rules_path = RESULTS_DIR / f"{ex_id}.rules"
        with rules_path.open("w", encoding="utf-8") as rf:
            rf.write(generation.openhab_code)

        # Deploy each rule to openHAB via MCP (only when valid and not disabled)
        deployed: List[str] = []
        if is_valid and not no_deploy:
            parser = RuleParser()
            parsed_rules = parser.parse_rules_text(generation.openhab_code)
            for rule in parsed_rules:
                destination_name = f"glint_{ex_id}_{rule.name}" if rule.name else f"glint_{ex_id}"
                maybe_deploy_via_mcp(
                    rule.raw_text,
                    request=text,
                    destination_name=destination_name,
                )
                deployed.append(destination_name)
        else:
            print(f"  Skipping MCP deployment for id={ex_id} (validation did not pass).")

        result_entry: Dict[str, Any] = {
            "id": ex_id,
            "request": text,
            "openhab_code": generation.openhab_code,
            "is_valid": is_valid,
            "validator_summary": validation.summary,
            "rules_path": str(rules_path),
            "attempts_used": attempts_used,
            "deployed_rules": deployed,
        }
        results.append(result_entry)

        # Write incrementally so progress is never lost on interruption
        with summary_path.open("w", encoding="utf-8") as sf:
            json.dump(results, sf, indent=2)

    print(f"\nWrote {len(results)} agent pipeline results to {RESULTS_DIR}")
    print(f"Summary JSON: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full agent workflow (with MCP context) on the Glint SmartThings dataset."
    )
    parser.add_argument(
        "--id",
        type=int,
        dest="only_id",
        help="If provided, run only for this dataset id.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=13.0,
        help=(
            "Seconds to wait between examples to respect RPM limits "
            "(default: 13s ≈ 4.6 RPM, safe for the free tier's 5 RPM cap). "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--no-deploy",
        dest="no_deploy",
        action="store_true",
        help="Skip MCP deployment even for valid rules (generate and validate only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_agent_pipeline_on_glint_dataset(
        only_id=args.only_id,
        delay=args.delay,
        no_deploy=args.no_deploy,
    )


if __name__ == "__main__":
    main()
