"""Prompt generation: fill templates with country pairs in both directions."""

from __future__ import annotations

from config import SCENARIOS, ALL_PAIRS


def make_prompt_id(scenario: str, country_a: str, country_b: str) -> str:
    """Deterministic ID: scenario__countryA_vs_countryB."""
    return f"{scenario}__{country_a}_vs_{country_b}"


def fill_template(template: str, country_a: str, country_b: str) -> str:
    return template.replace("[COUNTRY_A]", country_a).replace("[COUNTRY_B]", country_b)


def generate_all_prompts(
    pairs: list[tuple[str, str]] | None = None,
    scenarios: dict[str, str] | None = None,
) -> list[dict]:
    """Return list of prompt dicts with both direction variants.

    Each dict:
        prompt_id, scenario, country_a, country_b, direction, text
    """
    pairs = pairs or ALL_PAIRS
    scenarios = scenarios or SCENARIOS

    prompts = []
    for c1, c2 in pairs:
        for scen_name, template in scenarios.items():
            # Forward: A=c1, B=c2
            prompts.append({
                "prompt_id": make_prompt_id(scen_name, c1, c2),
                "scenario": scen_name,
                "country_a": c1,
                "country_b": c2,
                "direction": "forward",
                "pair": (c1, c2),
                "text": fill_template(template, c1, c2),
            })
            # Reverse: A=c2, B=c1
            prompts.append({
                "prompt_id": make_prompt_id(scen_name, c2, c1),
                "scenario": scen_name,
                "country_a": c2,
                "country_b": c1,
                "direction": "reverse",
                "pair": (c1, c2),
                "text": fill_template(template, c2, c1),
            })
    return prompts


if __name__ == "__main__":
    all_p = generate_all_prompts()
    print(f"Total prompts: {len(all_p)}")
    print(f"\nExample prompt ({all_p[0]['prompt_id']}):\n")
    print(all_p[0]["text"])
