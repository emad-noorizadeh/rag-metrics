#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate labeled (Q, answer, contexts) examples via an LLM prompt that includes
diverse seed examples. You control randomness (seed) and sampling (temperature).
We randomly choose 1 of 3 seed variants per category to keep each run fresh.

Outputs:
  - JSON (list of dicts with fields: q, a, c, y, reason)
  - CSV  (flattened)

Usage:
  python demo/generate_dataset.py --out_json demo/generated.json --out_csv demo/generated.csv \
    --temperature 0.7 --seed 42 --num_new 5

Integration:
  - This script does NOT call any vendor API by default. Fill in `call_llm()` for your provider.
  - You can keep using metric_utils + classifier on the generated data.
"""

from __future__ import annotations
import argparse, json, csv, random, textwrap, re
from typing import List, Dict, Any, Optional, Tuple
import openai
from openai import OpenAI

from dotenv import load_dotenv
_ = load_dotenv()

# -----------------------------
# 0) LLM call placeholder
# -----------------------------
def call_llm(prompt: str, model: str = "gpt-4o", temperature: float = 0.7) -> str:
    """
    Replace with your vendor/client code. Return the raw string the model outputs.
    Example (OpenAI):
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role":"user","content": prompt}]
        )
        return resp.choices[0].message.content
    """
    client = OpenAI()
    resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role":"user","content": prompt}]
        )
    return resp.choices[0].message.content


# -----------------------------
# 1) Seed pool (10 categories × 3)
# -----------------------------
# Each seed: {q, a, c, y, reason} where:
#   q: question
#   a: answer/response
#   c: list[str] contexts
#   y: 1 faithful / 0 unfaithful
#   reason: brief justification to help the model generalize good patterns
#
# Contexts are intentionally longer, with entities/dates/numbers to stress grounding.

SEED_POOL: Dict[str, List[Dict[str, Any]]] = {
    # 1) Finance: Revenue & Guidance
    "finance_revenue_guidance": [
        {
            "q": "What was the 2023 revenue and growth guidance?",
            "a": "Revenue was $5.2B and management guided for 12% growth.",
            "c": [
                "The company reported FY2023 revenue of $5.2B, up from $4.9B in FY2022. "
                "In the guidance section of the earnings call, management projected approximately "
                "twelve percent growth for the next fiscal year, citing improved conversion and retention.",

                "Analyst commentary from Q4 noted that the 2023 revenue outperformed initial expectations. "
                "Prepared remarks also emphasized that expansion into new verticals is expected to drive growth "
                "in the 10–14% range, with the midpoint at 12%."
            ],
            "y": 1,
            "reason": "Both money and percent match and are grounded explicitly in the context."
        },
        {
            "q": "What was the 2023 revenue and growth guidance?",
            "a": "Revenue was $5.4B and growth guidance was 15%.",
            "c": [
                "FY2023 revenue printed at $5.2B according to the 10-K filed last week. "
                "The outlook pointed to growth in the low-double digits, with a midpoint near 12%.",

                "In the CEO letter, they reiterated confidence in double-digit expansion, noting cost discipline "
                "and new product uptake as tailwinds."
            ],
            "y": 0,
            "reason": "Numbers do not match: answer says $5.4B and 15% vs. $5.2B and ~12% in context."
        },
        {
            "q": "What did they post for 2023 revenue and what growth did they guide to?",
            "a": "They posted $5.2B revenue and guided 12% growth.",
            "c": [
                "Section 2.1: Financial Highlights — Total revenue in 2023 reached $5.2B, reflecting 6% year-over-year growth. "
                "The CFO commentary calls for 10–14% expansion next year, guiding to about 12% at midpoint.",

                "Appendix: Supplemental Tables — Revenue by region confirms the consolidated $5.2B total. "
                "The guidance also mentions headwinds from FX, keeping the range conservative."
            ],
            "y": 1,
            "reason": "Exact match to revenue and midpoint guidance."
        },
    ],

    # 2) Finance: Gross Margin
    "finance_gross_margin": [
        {
            "q": "What is Q4 gross margin?",
            "a": "Q4 gross margin was 45%.",
            "c": [
                "Quarterly Summary: Q4 gross margin expanded to 45% due to better supply pricing and "
                "a richer mix of enterprise deals. Inventory write-downs impacted Q3 but were not repeated.",

                "Full-Year Overview: Gross margin trend shows gradual improvement from 42% in Q1 to "
                "45% in Q4."
            ],
            "y": 1,
            "reason": "Matches exactly the figure in the context."
        },
        {
            "q": "What is Q4 gross margin?",
            "a": "Q4 gross margin was 47%.",
            "c": [
                "Results: The company reported stable gross margins throughout the year. In Q4, margins were reported at 45%, "
                "reflecting favorable costs and product mix improvements.",

                "Outlook: Management expects margins to remain between 44% and 46% next quarter."
            ],
            "y": 0,
            "reason": "Answer says 47% but context states 45%."
        },
        {
            "q": "Could you tell me Q4 gross margin?",
            "a": "It came in at 45 percent.",
            "c": [
                "Press Release (Q4): Gross margin came in at 45% versus 43% in the prior quarter, driven by lower freight costs.",

                "Call Transcript: CFO reiterated a structural margin improvement with an expected band of 44–46% near-term."
            ],
            "y": 1,
            "reason": "Matches the reported value."
        },
    ],

    # 3) Dates: Event timing
    "dates_event": [
        {
            "q": "When did the kickoff event take place?",
            "a": "It took place on Jan 3, 2024.",
            "c": [
                "Internal Memo: The kickoff event occurred on Jan 3, 2024 at 10:30 local time in the main auditorium. "
                "Attendance was over 300 employees and the guest speaker joined remotely.",

                "Follow-up email on Jan 4, 2024 thanked attendees and shared links to recordings."
            ],
            "y": 1,
            "reason": "Exact date is present in the context."
        },
        {
            "q": "When did the kickoff event take place?",
            "a": "It took place on Feb 2, 2024.",
            "c": [
                "Meeting Notes: The kickoff event was held on Jan 3, 2024 at HQ, with workshops continuing through January 5.",

                "Calendar invites show the initial keynote on January 3 and breakout sessions the next two days."
            ],
            "y": 0,
            "reason": "Date does not match the context."
        },
        {
            "q": "Please share the kickoff event date.",
            "a": "January 3, 2024.",
            "c": [
                "Agenda: Kickoff keynote — Jan 3, 2024. Team workshops — Jan 4–5. Social — evening of Jan 4.",
                "Photos posted on the internal portal the following week."
            ],
            "y": 1,
            "reason": "Correctly aligned to the stated date."
        },
    ],

    # 4) Systems: Database engine
    "systems_db_engine": [
        {
            "q": "Which database engine powers the service?",
            "a": "The service runs on PostgreSQL.",
            "c": [
                "Architecture Overview: The transactional datastore is PostgreSQL 14 hosted on managed instances. "
                "Read replicas serve analytics staging, while the final analytics layer is in BigQuery.",

                "Operations SOP: Routine vacuuming schedules and backup cycles are tuned for OLTP workloads."
            ],
            "y": 1,
            "reason": "Matches the explicitly stated engine."
        },
        {
            "q": "Which database engine powers the service?",
            "a": "The service runs on MongoDB.",
            "c": [
                "Database Stack: We use PostgreSQL for transactions and Redis for caching. "
                "Analytics pipelines land in BigQuery for downstream analysis.",

                "Disaster Recovery: WAL archiving and PITR configured at the cluster level."
            ],
            "y": 0,
            "reason": "Answer contradicts the context."
        },
        {
            "q": "What DB engine underpins the service?",
            "a": "PostgreSQL 14 handles transactional storage.",
            "c": [
                "Technical Design Doc v2: Postgres 14 primary with 2 replicas. "
                "Latency-sensitive features rely on optimized indexes and partitioning.",

                "Analytics Architecture: Extractors read from read replicas into staging tables."
            ],
            "y": 1,
            "reason": "Aligned with the context (PostgreSQL 14)."
        },
    ],

    # 5) Hardware: Battery & Range
    "hardware_specs": [
        {
            "q": "What are the battery and range specs?",
            "a": "Battery is 500 mAh and range is 10 km.",
            "c": [
                "Specs Table: Battery capacity 500 mAh (typ.), nominal voltage 3.7V. Estimated travel range 10 km under eco mode.",

                "Field Notes: In colder conditions, range may decrease by ~10–15% due to lithium chemistry behavior."
            ],
            "y": 1,
            "reason": "Both quantitative values match."
        },
        {
            "q": "What are the battery and range specs?",
            "a": "Battery is 700 mAh and range is 12 km.",
            "c": [
                "Engineering BOM lists battery capacity of 500 mAh. Range measured during certification was 10 km on average.",

                "User Guide: Range depends on payload and terrain; nominal measurements established at 10 km."
            ],
            "y": 0,
            "reason": "Both quantities differ from the context."
        },
        {
            "q": "Share the battery capacity and expected range.",
            "a": "500 mAh battery, ~10 km range.",
            "c": [
                "Device Sheet v1.3: Capacity 500 mAh. Typical range: 10 km, tested at 18–20°C on flat routes.",

                "FAQ: Performance varies; battery health declines after ~300 cycles."
            ],
            "y": 1,
            "reason": "Consistent with the source numbers."
        },
    ],

    # 6) Pricing Plans
    "pricing_plans": [
        {
            "q": "Summarize the pricing.",
            "a": "Plan A is $20 and Plan B is $50.",
            "c": [
                "Pricing Page: Plan A — $20/month; Plan B — $50/month. Discounts apply for annual prepay.",

                "Support Policy: Email support for all plans; phone support only for enterprise."
            ],
            "y": 1,
            "reason": "Accurate pricing with no extra claims."
        },
        {
            "q": "Summarize the pricing.",
            "a": "Plan A is $20, Plan B is $50, and both include phone support.",
            "c": [
                "Public Pricing: Plan A — $20; Plan B — $50. Support matrix shows email support for both. "
                "Phone support requires enterprise plan.",

                "Change log: Phone support was removed from self-serve tiers last year."
            ],
            "y": 0,
            "reason": "Introduces unsupported claim about phone support."
        },
        {
            "q": "What are the plan prices?",
            "a": "Plan A is $20 and Plan B is $50 per month.",
            "c": [
                "Plan Catalog v2: A — $20; B — $50; annual discount applies.",

                "Support remains email-only for A/B; enterprise adds phone SLAs."
            ],
            "y": 1,
            "reason": "Matches context; no hallucinated features."
        },
    ],

    # 7) Growth Forecast
    "growth_forecast": [
        {
            "q": "What’s the forecast growth?",
            "a": "They projected twelve percent growth.",
            "c": [
                "Outlook: Management expects growth of 12 percent next year based on pipeline and retention improvements.",

                "Analyst summary: Forecast range 10–14%; midpoint 12%."
            ],
            "y": 1,
            "reason": "Textual variant of the 12% figure; still grounded."
        },
        {
            "q": "What’s the forecast growth?",
            "a": "They projected double-digit growth of 25%.",
            "c": [
                "Outlook: Management expects growth of 12 percent next year. "
                "They highlighted caution around FX and macro.",

                "Notes: The CFO reiterated a mid-teens ceiling only if upside materializes; baseline remains ~12%."
            ],
            "y": 0,
            "reason": "25% contradicts the stated 12%."
        },
        {
            "q": "What’s the expected growth rate?",
            "a": "Approximately 12% growth next year.",
            "c": [
                "Q&A: Guidance reiterates ~12% growth; upside drivers are new logos and pricing initiatives.",

                "Risks: FX, competitive pressure; range 10–14%."
            ],
            "y": 1,
            "reason": "Matches midpoint guidance."
        },
    ],

    # 8) SDK Language
    "sdk_language": [
        {
            "q": "Which language is the SDK in?",
            "a": "The SDK is in Python.",
            "c": [
                "Developer Docs: The primary SDK is provided in Python and maintained by the platform team. "
                "TypeScript bindings are available but experimental.",

                "Release Notes: Python SDK v2 adds streaming APIs."
            ],
            "y": 1,
            "reason": "Fully grounded in docs."
        },
        {
            "q": "Which language is the SDK in?",
            "a": "The SDK is in Python and includes a Rust extension.",
            "c": [
                "Public Docs: SDK is in Python. There is no mention of Rust extensions in the official repo.",

                "Changelog: The last release focused on Python packaging fixes and docs."
            ],
            "y": 0,
            "reason": "Rust extension is not supported by the context."
        },
        {
            "q": "What language is the SDK available in?",
            "a": "Python.",
            "c": [
                "Quickstart: Installation via pip; the Python package name is listed with usage examples.",

                "API Reference: Python methods with code snippets; TS bindings noted as preview."
            ],
            "y": 1,
            "reason": "Concise and faithful."
        },
    ],

    # 9) Policy Eligibility (non-numeric)
    "policy_eligibility": [
        {
            "q": "Who qualifies for the benefit?",
            "a": "Full-time employees with 6 months tenure; contractors are excluded.",
            "c": [
                "Policy v3.2: Eligibility — full-time employees after 6 months continuous service. "
                "Contractors, interns, and part-time staff are not eligible.",

                "HR FAQ: Exceptions require VP approval; rare."
            ],
            "y": 1,
            "reason": "Matches the rule and exclusions."
        },
        {
            "q": "Who qualifies for the benefit?",
            "a": "All full-time employees and contractors are eligible immediately.",
            "c": [
                "Eligibility: Full-time after 6 months. Contractors and interns are excluded unless explicitly approved.",

                "Benefits Portal: A waiting period applies to new hires."
            ],
            "y": 0,
            "reason": "Contradicts timing and includes excluded roles."
        },
        {
            "q": "Who can receive the stipend?",
            "a": "Full-time staff with 6 months tenure; no contractors.",
            "c": [
                "Stipend Rule: Applies to full-time after 6 months. Contractors cannot participate.",

                "FAQ: Requests for exception need VP sign-off."
            ],
            "y": 1,
            "reason": "Aligned with the documented policy."
        },
    ],

    # 10) Medical dosage / units (QUANTITY)
    "medical_dosage": [
        {
            "q": "What is the recommended adult dosage and schedule?",
            "a": "500 mg twice daily for 7 days.",
            "c": [
                "Guideline: Adult dosing — 500 mg twice daily (BID) for 7 days in uncomplicated cases. "
                "Pediatric dosing differs by weight.",

                "Notes: Take with food to minimize GI side effects."
            ],
            "y": 1,
            "reason": "Exact dose, frequency, and duration match."
        },
        {
            "q": "What is the recommended adult dosage and schedule?",
            "a": "750 mg once daily for 5 days.",
            "c": [
                "Guideline: Adult dosing — 500 mg twice daily for 7 days. "
                "Alternative regimens are not recommended for standard cases.",

                "Cautions: Adjust for renal impairment."
            ],
            "y": 0,
            "reason": "Dose, frequency, and duration do not match."
        },
        {
            "q": "What is the adult regimen?",
            "a": "500 mg two times per day for a full week.",
            "c": [
                "Protocol: Adult dose — 500 mg BID × 7 days. Ensure adherence.",
                "Adverse Events: Rare; monitor GI symptoms."
            ],
            "y": 1,
            "reason": "Paraphrase remains faithful."
        },
    ],
}

CATEGORIES = list(SEED_POOL.keys())


# -----------------------------
# 2) Build prompt with samples
# -----------------------------
def pick_seed_set(rng: random.Random) -> List[Dict[str, Any]]:
    selected = []
    for cat in CATEGORIES:
        variants = SEED_POOL[cat]
        selected.append(rng.choice(variants))
    return selected

def build_prompt(seeds: List[Dict[str, Any]], num_new: int = 6) -> str:
    # enforce even split: half faithful, half unfaithful
    half = num_new // 2

    instr = textwrap.dedent(f"""
    You are generating labeled RAG evaluation items.

    TASK:
      Produce exactly {num_new} NEW examples that are diverse and *not* trivial copies of seeds.
      Output must be a JSON array of {num_new} items.
      Each item has fields:
        - "q": string   (question)
        - "a": string   (answer/response)
        - "c": string[] (list of 2–5 context passages, realistic multi-sentence)
        - "y": 0 or 1   (label: 1 = faithful/grounded/correct, 0 = unfaithful/unsupported/incorrect)
        - "reason": short string explanation for the label

    BALANCE REQUIREMENT:
      - Exactly {half} examples must have y=1 (faithful/grounded).
      - Exactly {half} examples must have y=0 (unfaithful/unsupported).

    RULES:
      - Use grounded facts in contexts to justify labels.
      - Include numeric entities, dates, units, or policy rules in some items so faithfulness can be tested.
      - Negative (y=0) cases must vary: wrong numbers, unsupported claims, contradicted policies, hallucinated extras.
      - Contexts must be realistic passages (2–5 items, each a few sentences).
      - Return ONLY the JSON array. No extra commentary.

    SEEDS (for diversity pattern; DO NOT copy verbatim):
    """).strip()

    seed_blocks = []
    for i, s in enumerate(seeds, start=1):
        seed_blocks.append(textwrap.dedent(f"""
        // Seed {i}
        {{
          "q": {json.dumps(s["q"])},
          "a": {json.dumps(s["a"])},
          "c": {json.dumps(s["c"])},
          "y": {s["y"]},
          "reason": {json.dumps(s["reason"])}
        }}
        """).strip())

    prompt = instr + "\n[\n" + ",\n".join(seed_blocks) + "\n]\n"
    prompt += f"\nNOW RETURN ONLY A JSON ARRAY OF {num_new} NEW ITEMS WITH EXACTLY {half} y=1 AND {half} y=0:"
    return prompt

# -----------------------------
# 3) Output repair + validation
# -----------------------------
def repair_to_json(raw: str) -> List[Dict[str, Any]]:
    """
    Try to coerce the LLM output into a JSON list of objects.
    Strategy:
      - Grab first [... ] block
      - If that fails, try to find ```json ... ``` fenced block
      - Fall back to best-effort bracket slice
    """
    # fenced code block
    m = re.search(r"```json\s*($begin:math:display$[\\s\\S]*?$end:math:display$)\s*```", raw, re.IGNORECASE)
    if m:
        raw = m.group(1).strip()

    # direct array
    if not raw.strip().startswith("["):
        # try to find first [ ... last ]
        start = raw.find("[")
        end = raw.rfind("]")
        if start >= 0 and end > start:
            raw = raw[start:end+1]

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # last resort: try to replace invalid trailing commas
    try:
        raw2 = re.sub(r",\s*([\]}])", r"\1", raw)
        data = json.loads(raw2)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    raise ValueError("Could not parse LLM output as a JSON array.")


def validate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for it in items:
        q = it.get("q"); a = it.get("a"); c = it.get("c"); y = it.get("y")
        if not isinstance(q, str) or not isinstance(a, str):
            continue
        if not isinstance(c, list) or not all(isinstance(s, str) for s in c):
            continue
        if not isinstance(y, int) or y not in (0, 1):
            continue
        # keep reason if present, else synthesize
        if "reason" not in it or not isinstance(it["reason"], str):
            it["reason"] = "n/a"
        out.append({"q": q, "a": a, "c": c, "y": y, "reason": it["reason"]})
    return out


# -----------------------------
# 4) Save helpers
# -----------------------------
def save_json(items: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def save_csv(items: List[Dict[str, Any]], path: str) -> None:
    # flatten contexts into a single string per row
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q", "a", "c_joined", "y", "reason"])
        for it in items:
            c_joined = "\n---\n".join(it["c"])
            w.writerow([it["q"], it["a"], c_joined, it["y"], it["reason"]])


# -----------------------------
# 5) Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", type=str, default="demo/generated.json")
    ap.add_argument("--out_csv", type=str, default="demo/generated.csv")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_new", type=int, default=5)
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # 1) Pick 1 of 3 variants from each category (10 categories × 1 = 10 seeds)
    seeds = pick_seed_set(rng)

    # 2) Build prompt
    prompt = build_prompt(seeds, num_new=args.num_new)

    # 3) Call your LLM (wire up call_llm() to your provider)
    try:
        raw = call_llm(prompt, model=args.model, temperature=args.temperature)
    except NotImplementedError:
        print("call_llm() is not implemented. Showing the prompt so you can test manually:\n")
        print(prompt)
        return

    # 4) Repair + validate
    data = repair_to_json(raw)
    data = validate_items(data)

    # 5) Append fixed text *into* every context entry
    FIXED_TEXTS = [
        "Preferred Rewards Platinum tier members using a Bank of America Debit or ATM card are not charged "
        "the non-Bank of America ATM fee for one withdrawal and one transfer per statement cycle from a "
        "non-Bank of America ATM in the U.S., and receive a refund of the ATM operator fee for one withdrawal "
        "and one transfer per statement cycle from a non-Bank of America ATM in the U.S. Preferred Rewards "
        "Platinum Honors and Diamond Honors tier members using a Bank of America Debit or ATM card are not "
        "charged the non-Bank of America ATM fee for withdrawals and transfers from non-Bank of America ATMs "
        "in the U.S. and U.S. territories and receive a refund of the ATM operator fee for withdrawals and "
        "transfers from non-Bank of America ATMs in the U.S.",

        "Preferred Rewards members who apply for an Auto purchase or refinance loan receive an interest rate "
        "discount of 0.25% for Gold tier, 0.35% for Platinum tier, and 0.50% for Platinum Honors and higher "
        "based on their Preferred Rewards tier at the time of auto loan application. The maximum Preferred "
        "Rewards interest rate discount on a Bank of America auto loan is 0.50%. This interest rate discount "
        "is not reflected in all our published rates on our website but will be confirmed and reflected in "
        "the interest rate quoted upon loan approval. Discounts are only available on auto loan applications "
        "submitted by you directly to Bank of America through its website, Financial Centers, or Bank call "
        "centers. Discounts are not available for motor vehicle leases or for applications sourced from car "
        "dealerships, car manufacturers, or third-party branded/co-branded relationships. Benefit is "
        "non-transferable. Subject to credit approval. Standard underwriting guidelines and credit policies apply."
    ]

    fixed_passage = rng.choice(FIXED_TEXTS)

    for item in data:
        if "c" in item and isinstance(item["c"], list):
            item["c"] = [f"{ctx.strip()} {fixed_passage}" for ctx in item["c"]]

    # 6) Save
    save_json(data, args.out_json)
    save_csv(data, args.out_csv)

    print(f"Saved {len(data)} items → {args.out_json} and {args.out_csv}")


if __name__ == "__main__":
    main()