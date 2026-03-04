from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

from agents.policy_generator import GenerationResult, PolicyGeneratorAgent
from agents.validator_agent import ValidationResult, ValidatorAgent
from agents.context_validator import ContextValidatorAgent
from tools.context_loader import load_contexts
from tools.context_fetcher import SystemContextFetcher, SystemContext
from tools.loader import save_rule
from tools.mcp_client import deploy_rule_via_mcp
from tools.prompt_builder import PromptBuilder
from tools.rule_parser import RuleParser


def sanitize_filename(name: str) -> str:
    """Convert a rule name to a safe filename."""
    # Remove quotes and special characters, replace spaces with underscores
    safe_name = name.replace('"', '').replace("'", '')
    safe_name = ''.join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in safe_name)
    safe_name = safe_name.replace(' ', '_').lower()
    # Remove consecutive underscores
    while '__' in safe_name:
        safe_name = safe_name.replace('__', '_')
    return safe_name.strip('_')


def save_rules_individually(code: str, prefix: str = None) -> list[str]:
    """Parse generated code and save each rule to a separate file.
    
    Returns list of saved file paths.
    """
    parser = RuleParser()
    rules = parser.parse_rules_text(code)
    
    if not rules:
        # No rules found, save as single file
        filename = f"{prefix}.rules" if prefix else "generated.rules"
        path = save_rule(code, filename=filename)
        print(f"No parseable rules found. Saved raw code to {path}")
        return [path]
    
    saved_paths = []
    for i, rule in enumerate(rules, 1):
        # Generate filename from rule name
        base_name = sanitize_filename(rule.name)
        if not base_name:
            base_name = f"rule_{i}"
        
        # Add prefix if provided
        if prefix:
            filename = f"{prefix}_{base_name}.rules"
        else:
            filename = f"{base_name}.rules"
        
        # Save the individual rule
        path = save_rule(rule.raw_text, filename=filename)
        print(f"Saved rule '{rule.name}' to {path}")
        saved_paths.append(path)
    
    return saved_paths


def run_generation_loop(
    request: str,
    retriever,
    *,
    max_attempts: int,
    system_context: SystemContext | None = None,
) -> Tuple[GenerationResult, ValidationResult, bool, int]:
    """Iteratively generate and validate openHAB code with a retry guard.

    Returns:
        generation: Final generation result (from the last attempt).
        validation: Validation result associated with the returned generation.
        is_valid: True if a syntactically (and contextually) valid rule was found.
        attempts_used: How many generation/validation attempts were performed.
    """
    docs = retriever.invoke(request)
    prompt_builder = PromptBuilder(request=request, documents=list(docs))
    
    # Set system context if available
    if system_context:
        prompt_builder.set_system_context(system_context)
    
    generator = PolicyGeneratorAgent()
    validator = ValidatorAgent()
    context_validator = ContextValidatorAgent() if system_context else None

    last_validation: ValidationResult | None = None
    last_generation: GenerationResult | None = None
    for attempt in range(1, max_attempts + 1):
        print(f"\n=== Generation attempt {attempt}/{max_attempts} ===")
        generation = generator.generate(**prompt_builder.generator_variables())
        prompt_builder.record_candidate(generation.openhab_code)
        
        # Syntax validation first (no point checking live system if syntax is broken)
        print("Running syntax validation...")
        validation = validator.validate(**prompt_builder.validator_variables(generation.openhab_code))
        last_generation = generation

        if not validation.is_valid:
            print("Syntax Validator: FAIL")
            last_validation = validation
            feedback_entry = validation.as_feedback_entry()
            prompt_builder.add_feedback(source=f"syntax_validator attempt {attempt}", message=feedback_entry)
            print(feedback_entry)
            continue
        
        print("Syntax Validator: PASS")
        
        # Context-aware validation (only if syntax is valid)
        if context_validator and system_context:
            print("Running context validation...")
            context_validation = context_validator.validate(
                candidate_code=generation.openhab_code,
                system_context=system_context,
                request=request,
            )
            
            if not context_validation.is_valid:
                print("Context Validator: FAIL")
                print(context_validation.as_feedback_entry())
                prompt_builder.add_feedback(
                    source=f"context_validator attempt {attempt}",
                    message=context_validation.as_feedback_entry()
                )
                # Keep syntax validation so we can return (generation, validation, False) instead of raising
                last_validation = validation
                continue
            else:
                print("Context Validator: PASS")
                if context_validation.warnings:
                    print("Warnings:", context_validation.warnings)
        
        # All validations passed!
        return generation, validation, True, attempt

    if last_generation is None or last_validation is None:
        raise RuntimeError("Generation loop terminated without producing any attempts.")

    print(
        f"Exceeded {max_attempts} attempts without validator approval. "
        "Saving latest result for manual review."
    )
    # If we never achieved a valid rule, we conservatively report that all
    # configured attempts were used.
    return last_generation, last_validation, False, max_attempts


def maybe_deploy_via_mcp(rule_code: str, *, request: str, destination_name: str) -> None:
    """Attempt to deploy the validated rule to openHAB via MCP protocol."""
    # Check if deployment is disabled
    if os.environ.get("DISABLE_MCP_DEPLOYMENT", "").lower() == "true":
        print("MCP deployment disabled via DISABLE_MCP_DEPLOYMENT.")
        return

    print(f"Deploying rule '{destination_name}' to openHAB via MCP...")
    success, message = deploy_rule_via_mcp(
        rule_code,
        rule_name=destination_name,
        metadata={"request": request},
    )
    status = "succeeded" if success else "failed"
    print(f"MCP deployment {status}: {message}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="iconpal-openhab: NL -> openHAB code")
    parser.add_argument("prompt", type=str, nargs="*", help="Natural language request")
    parser.add_argument("--out", dest="out", type=str, default=None, help="Output filename prefix (each rule saved to separate .rules file)")
    parser.add_argument(
        "--max-attempts",
        dest="max_attempts",
        type=int,
        default=int(os.environ.get("GENERATION_MAX_ATTEMPTS", "3")),
        help="Maximum generator/validator iterations (default: 3 or GENERATION_MAX_ATTEMPTS).",
    )
    parser.add_argument(
        "--no-context-validation",
        dest="no_context_validation",
        action="store_true",
        help="Disable context-aware validation (skip live system checks)",
    )
    args = parser.parse_args()

    if not args.prompt:
        print("Please provide a natural language request.")
        sys.exit(1)

    request = " ".join(args.prompt).strip()
    retriever = load_contexts(path="./context", vs_path="./vectorstore/faiss")

    # Fetch live system context (if configured and not disabled)
    system_context = None
    enable_context = os.environ.get("ENABLE_CONTEXT_VALIDATION", "true").lower() != "false"

    if enable_context and not args.no_context_validation:
        print(f"\n=== Fetching live system context via MCP server ===")
        try:
            # SystemContextFetcher will launch/connect to the openHAB MCP server,
            # which in turn talks to your openHAB instance using environment config.
            context_fetcher = SystemContextFetcher()
            system_context = context_fetcher.fetch_all()
            print("✓ System context loaded successfully\n")
        except Exception as e:
            print(f"⚠ Warning: Could not fetch system context: {e}")
            print("Proceeding without context-aware validation.\n")

    generation, validation, is_valid, attempts_used = run_generation_loop(
        request,
        retriever,
        max_attempts=args.max_attempts,
        system_context=system_context,
    )

    # Use --out as a prefix for multiple rules, or None for auto-naming
    prefix = Path(args.out).stem if args.out else None
    saved_paths = save_rules_individually(generation.openhab_code, prefix=prefix)
    
    print(f"\n=== Summary ===")
    print(f"Generated {len(saved_paths)} rule file(s)")
    print(f"Validator summary: {validation.summary}")
    print(f"Generation attempts used: {attempts_used}")
    if not is_valid:
        print(validation.as_feedback_entry())

    # Deploy each rule via MCP if valid
    if is_valid:
        parser = RuleParser()
        rules = parser.parse_rules_text(generation.openhab_code)
        for rule, path in zip(rules, saved_paths):
            destination_name = Path(path).stem
            maybe_deploy_via_mcp(rule.raw_text, request=request, destination_name=destination_name)
    else:
        print("Skipping MCP deployment because validator did not approve the rule.")


if __name__ == "__main__":
    main()

