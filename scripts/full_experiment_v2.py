"""
Full V2 Experiment: Personality change after life events

Complete experiment:
- 100 personas: 2 genders × 5 continents × 10 personality types
- 11 life events
- BFI-44 measurement with single-call as the canonical default
- Model: configurable
"""

import json
import re
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import numpy as np
from collections import Counter
from scipy import stats

# Optional dependency: allow running without python-dotenv if env vars are set externally.
try:
    from dotenv import load_dotenv  # type: ignore
except ModuleNotFoundError:  # pragma: no cover

    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False


# Load environment variables
env_file = os.getenv("ENV_FILE", ".env")
load_dotenv(env_file)

# ============================================================
# Persona Configuration
# ============================================================

GENDERS = ["Male", "Female"]
CONTINENTS = ["Europe", "Americas", "Africa", "Asia", "Oceania"]


def _implicit_controlled_profile(profile_id: str) -> str:
    """Return an implicit persona description.

    Requirement: do NOT explicitly name personality dimensions or levels.
    The text should *implicitly* express one strong tendency while keeping other aspects relatively balanced.
    """

    profiles = {
        # Extraversion high / low
        "E_high": (
            "Naturally sociable and energetic. You enjoy conversation, often take initiative in groups, "
            "and feel recharged by being around people. You usually keep a steady emotional baseline, "
            "can be considerate without being a pushover, and are organized enough to follow through "
            "without becoming rigid or perfectionistic. You’re curious and open to new ideas, but you "
            "also stay grounded and practical."
        ),
        "E_low": (
            "Quiet and reserved by default. You prefer small groups or one-on-one interactions, and you "
            "don’t seek attention. You’re generally even-tempered, reasonably considerate, and not prone "
            "to dramatic mood swings. You keep your life moderately organized—reliable but not obsessive. "
            "You can appreciate new ideas when they make sense, without chasing novelty for its own sake."
        ),
        # Agreeableness high / low
        "A_high": (
            "Warm and cooperative. You try to see the good in people, avoid unnecessary conflict, and "
            "often look for compromises. You stay fairly emotionally steady, keep your routines moderately "
            "structured, and are neither impulsively adventurous nor strongly resistant to change. You can "
            "be sociable when needed but don’t constantly seek the spotlight."
        ),
        "A_low": (
            "Direct, skeptical, and hard to impress. You value honesty over harmony and you don’t sugarcoat "
            "opinions. You’re not constantly anxious or emotionally volatile—you can stay composed—but you "
            "tend to challenge people’s claims and push back when you disagree. Your day-to-day organization "
            "is average, and you’re not especially drawn to novelty or tradition; you judge things case by case."
        ),
        # Conscientiousness high / low
        "C_high": (
            "Disciplined and planful. You like clear goals, follow schedules, and feel satisfied when things "
            "are done properly. You’re generally calm rather than easily rattled, socially flexible without "
            "being overly talkative, and you can be kind without always prioritizing others over yourself. "
            "You’re open to reasonable change, but you prefer it to be purposeful rather than chaotic."
        ),
        "C_low": (
            "Spontaneous and flexible with structure. You often act on the moment, procrastinate occasionally, "
            "and dislike rigid routines. Your mood is usually stable enough (not constantly worried), and you "
            "can get along with people without being especially warm or especially combative. You’re moderately "
            "curious, but you don’t feel a constant need to reinvent everything."
        ),
        # Neuroticism high / low
        "N_high": (
            "Emotionally reactive and sensitive to stress. You notice worries quickly, replay mistakes, and can "
            "be easily unsettled by uncertainty. Despite that, you try to be reasonably fair with others, keep "
            "your life moderately organized, and you’re not inherently thrill-seeking or highly novelty-driven. "
            "Socially, you can engage when needed, but your energy often depends on how safe and predictable you feel."
        ),
        "N_low": (
            "Calm, steady, and difficult to rattle. You recover quickly from setbacks and rarely spiral into worry. "
            "You’re neither unusually strict nor unusually careless—your organization is moderate. You can be friendly "
            "without constantly seeking company, and you can be open-minded without being eccentric. You generally respond "
            "to problems with a practical, composed mindset."
        ),
        # Openness high / low
        "O_high": (
            "Imaginative and curious. You enjoy exploring new ideas, art, and alternative ways of doing things, and you "
            "tend to get interested in abstract questions. You still keep your responsibilities at a reasonable level—"
            "not chaotic, not overly rigid. You’re usually emotionally steady, socially adaptable without needing constant "
            "attention, and you can be considerate without always avoiding disagreement."
        ),
        "O_low": (
            "Practical and preference for the familiar. You like proven methods, clear rules, and concrete facts more than "
            "abstract theorizing. You keep your life moderately organized and your emotions fairly steady. You can cooperate "
            "with others without being overly accommodating, and you can be social when required without seeking constant stimulation."
        ),
    }

    if profile_id not in profiles:
        raise KeyError(f"Unknown implicit profile_id: {profile_id}")

    return profiles[profile_id]


PERSONALITY_DESCRIPTIONS = {
    # Extraversion
    "P1": _implicit_controlled_profile("E_high"),
    "P2": _implicit_controlled_profile("E_low"),
    # Agreeableness
    "P3": _implicit_controlled_profile("A_high"),
    "P4": _implicit_controlled_profile("A_low"),
    # Conscientiousness
    "P5": _implicit_controlled_profile("C_high"),
    "P6": _implicit_controlled_profile("C_low"),
    # Neuroticism
    "P7": _implicit_controlled_profile("N_high"),
    "P8": _implicit_controlled_profile("N_low"),
    # Openness
    "P9": _implicit_controlled_profile("O_high"),
    "P10": _implicit_controlled_profile("O_low"),
}

# ============================================================
# Life Events (all 11)
# ============================================================

LIFE_EVENTS = {
    # Occupational domain
    "graduation": {
        "domain": "occupational",
        "notification": "You've just graduated! Diploma in hand, ready for the next chapter.",
        "reflection_prompt": "How do you feel about graduating? What are your thoughts on this milestone?",
        "expected_changes": {"C": "+", "N": "-", "E": "-"},
    },
    "work_entry": {
        "domain": "occupational",
        "notification": "Today was your first day at your new job. Everything feels fresh and challenging.",
        "reflection_prompt": "How do you feel about starting this new job? What are your expectations?",
        "expected_changes": {"C": "+", "N": "-"},
    },
    "job_change": {
        "domain": "occupational",
        "notification": "You've just switched to a completely different career field. New colleagues, new skills.",
        "reflection_prompt": "How do you feel about this career change? What motivated you?",
        "expected_changes": {"E": "?", "O": "?"},
    },
    "promotion": {
        "domain": "occupational",
        "notification": "You've been promoted to a senior position with more responsibilities.",
        "reflection_prompt": "How do you feel about this promotion? What does it mean to you?",
        "expected_changes": {"C": "+", "N": "-", "O": "+"},
    },
    "unemployment": {
        "domain": "occupational",
        "notification": "You've been laid off. Your position was eliminated in a restructuring.",
        "reflection_prompt": "How are you coping with losing your job? What are your plans?",
        "expected_changes": {"C": "-", "O": "-", "A": "-"},
    },
    "retirement": {
        "domain": "occupational",
        "notification": "You've retired. Today was your last day at work after decades.",
        "reflection_prompt": "How do you feel about retiring? What will you do now?",
        "expected_changes": {"C": "-"},
    },
    # Social domain
    "new_relationship": {
        "domain": "social",
        "notification": "You've started dating someone special. Things are going well.",
        "reflection_prompt": "How do you feel about this new relationship? What do you hope for?",
        "expected_changes": {"N": "-", "E": "+", "A": "-"},
    },
    "marriage": {
        "domain": "social",
        "notification": "You just got married! You're beginning life together as a couple.",
        "reflection_prompt": "How do you feel about getting married? What changes do you anticipate?",
        "expected_changes": {"N": "-", "E": "-", "O": "-", "A": "-"},
    },
    "divorce": {
        "domain": "social",
        "notification": "You're getting divorced. You and your spouse have decided to separate.",
        "reflection_prompt": "How are you processing this divorce? What are your thoughts?",
        "expected_changes": {"O": "+", "A": "+"},
    },
    "child_birth": {
        "domain": "social",
        "notification": "Your baby was born today! You're now a parent.",
        "reflection_prompt": "How do you feel about becoming a parent? What are your hopes and fears?",
        "expected_changes": {"E": "-", "C": "-"},
    },
    # Health domain
    "chronic_illness": {
        "domain": "health",
        "notification": "You've been diagnosed with a chronic condition. It's manageable but lifelong.",
        "reflection_prompt": "How are you dealing with this diagnosis? What are your thoughts?",
        "expected_changes": {"N": "+", "E": "-", "O": "-", "C": "-"},
    },
}

# ============================================================
# BFI Item Loading
# ============================================================


@dataclass
class BFIItem:
    id: int
    statement: str
    trait: str
    reverse: bool


def load_bfi_items() -> Dict[int, BFIItem]:
    """Load BFI-44 items from BFI.json."""
    bfi_path = Path(__file__).parent.parent / "BFI.json"
    with open(bfi_path, "r", encoding="utf-8") as f:
        bfi_data = json.load(f)

    dimension_to_trait = {
        "Extraversion": "E",
        "Agreeableness": "A",
        "Conscientiousness": "C",
        "Neuroticism": "N",
        "Openness": "O",
    }

    reverse_items = set(bfi_data.get("reverse", []))

    items = {}
    for item_id_str, item_data in bfi_data["questions"].items():
        item_id = int(item_id_str)
        trait = dimension_to_trait.get(item_data["dimension"], "O")

        items[item_id] = BFIItem(
            id=item_id,
            statement=item_data["origin_en"],
            trait=trait,
            reverse=(item_id in reverse_items),
        )

    return items


BFI_ITEMS = load_bfi_items()

# ============================================================
# Prompt Templates
# ============================================================


def create_persona_system_prompt(
    gender: str, continent: str, personality_id: str
) -> str:
    """Create system prompt for descriptive persona."""
    personality_desc = PERSONALITY_DESCRIPTIONS.get(personality_id, "")

    return f"""You are role-playing as a person with the following characteristics:

- Gender: {gender}
- Cultural Background: {continent} (raised and living in {continent})
- Personality Description: {personality_desc}

Please answer the following questions as this person would, staying true to their personality traits and cultural background.
Do not break character or mention that you are an AI.
Respond naturally as this person would."""


def create_direct_question(item) -> str:
    """Create a direct question for a single BFI item (multi-call method)."""
    return f"""Please rate how much you agree with this statement about yourself:

"I am someone who {item.statement.lower()}"

Please respond with a number from 1-5:
1 - Disagree strongly
2 - Disagree a little
3 - Neutral; no opinion
4 - Agree a little
5 - Agree strongly

Please respond with only the number (1-5)."""


def create_single_call_prompt() -> str:
    """Create prompt for single-call measurement (all 44 questions at once)."""
    questions = []
    for item_id in sorted(BFI_ITEMS.keys()):
        item = BFI_ITEMS[item_id]
        questions.append(f"{item_id}. I am someone who {item.statement.lower()}")

    questions_text = "\n".join(questions)

    return f"""Please rate how much you agree with each statement about yourself on a scale of 1-5:
1 = Disagree strongly
2 = Disagree a little
3 = Neutral; no opinion
4 = Agree a little
5 = Agree strongly

Please respond with ONLY the item number and your rating, one per line.
Format: number. rating
Example:
1. 4
2. 2
...

Here are the statements:

{questions_text}"""


# ============================================================
# Score Functions
# ============================================================


def extract_single_call_scores(response: str) -> Dict[int, int]:
    """Extract scores from single-call response."""
    scores = {}

    # Anchor to full lines to avoid cross-line matches such as "4  \n5".
    patterns = [
        r"(?m)^\s*(\d+)\s*[\.:\)]\s*([1-5])\s*$",
        r"(?m)^\s*(\d+)\s*-\s*([1-5])\s*$",
        r"(?m)^\s*(\d+)\s+([1-5])\s*$",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            item_id = int(match[0])
            score = int(match[1])
            if 1 <= item_id <= 44 and 1 <= score <= 5:
                scores[item_id] = score

    return scores


def extract_direct_score(response: str) -> int:
    """Extract numeric score from direct answer response (multi-call method)."""
    match = re.search(r"\b([1-5])\b", response)
    return int(match.group(1)) if match else 3


def calculate_trait_scores(responses: Dict[int, int]) -> Dict[str, float]:
    """Calculate Big Five trait scores from item responses."""
    traits = {"E": [], "A": [], "C": [], "N": [], "O": []}

    for item_id, score in responses.items():
        if item_id not in BFI_ITEMS:
            continue
        item = BFI_ITEMS[item_id]
        trait = item.trait

        if item.reverse:
            score = 6 - score

        traits[trait].append(score)

    return {
        "Extraversion": sum(traits["E"]) / len(traits["E"]) if traits["E"] else 0,
        "Agreeableness": sum(traits["A"]) / len(traits["A"]) if traits["A"] else 0,
        "Conscientiousness": sum(traits["C"]) / len(traits["C"]) if traits["C"] else 0,
        "Neuroticism": sum(traits["N"]) / len(traits["N"]) if traits["N"] else 0,
        "Openness": sum(traits["O"]) / len(traits["O"]) if traits["O"] else 0,
    }


# ============================================================
# API Client
# ============================================================


def get_client():
    """Create OpenAI-compatible client."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm(
    client,
    messages: List[Dict],
    model: str,
    max_tokens: int = 2000,
    temperature: float = 0.7,
) -> str:
    """Call LLM with messages."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ============================================================
# Measurement (Single-Call)
# ============================================================


def measure_bfi_single_call(
    client,
    system_prompt: str,
    model: str,
    prior_messages: List[Dict] = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Dict:
    """Measure BFI using single-call method with optional conversation context."""
    user_prompt = create_single_call_prompt()

    messages = [{"role": "system", "content": system_prompt}]
    if prior_messages:
        messages.extend(prior_messages)
    messages.append({"role": "user", "content": user_prompt})

    # Ratings should be concise; temperature can be overridden for stochasticity tests.
    response = call_llm(
        client, messages, model, max_tokens=max_tokens, temperature=temperature
    )
    scores = extract_single_call_scores(response)
    trait_scores = calculate_trait_scores(scores)

    # Add assistant response to messages
    full_messages = messages.copy()
    full_messages.append({"role": "assistant", "content": response})

    return {
        "method": "single",
        "item_scores": scores,
        "trait_scores": trait_scores,
        "items_extracted": len(scores),
        "raw_response": response,
        "messages": full_messages,
    }


# ============================================================
# Measurement (Multi-Call)
# ============================================================


def measure_bfi_multi_call(
    client,
    system_prompt: str,
    model: str,
    prior_messages: List[Dict] = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 8,
) -> Dict:
    """Measure BFI using multi-call method with conversation context.

    Each BFI item is asked separately, and all messages are saved for each item.
    """
    scores = {}
    item_conversations = []  # Store full conversation for each item

    for item_id in sorted(BFI_ITEMS.keys()):
        item = BFI_ITEMS[item_id]
        user_prompt = create_direct_question(item)

        # Build messages with prior context
        messages = [{"role": "system", "content": system_prompt}]
        if prior_messages:
            messages.extend(prior_messages)
        messages.append({"role": "user", "content": user_prompt})

        # Multi-call answers are a single digit; keep it tight.
        response = call_llm(
            client, messages, model, max_tokens=max_tokens, temperature=temperature
        )
        score = extract_direct_score(response)
        scores[item_id] = score

        # Save full conversation for this item
        item_conversation = {
            "item_id": item_id,
            "statement": item.statement,
            "trait": item.trait,
            "reverse": item.reverse,
            "score": score,
            "raw_response": response,
            "messages": messages + [{"role": "assistant", "content": response}],
        }
        item_conversations.append(item_conversation)

        time.sleep(0.05)  # Small delay

    trait_scores = calculate_trait_scores(scores)

    return {
        "method": "multi",
        "item_scores": scores,
        "trait_scores": trait_scores,
        "items_extracted": len(scores),
        "item_conversations": item_conversations,
    }


# ============================================================
# Generate All Personas
# ============================================================


def generate_all_personas() -> List[Dict]:
    """Generate all 100 personas."""
    personas = []
    for gender in GENDERS:
        for continent in CONTINENTS:
            for pid in PERSONALITY_DESCRIPTIONS.keys():
                persona_id = f"{gender[0]}_{continent[:3]}_{pid}"
                personas.append(
                    {
                        "id": persona_id,
                        "gender": gender,
                        "continent": continent,
                        "personality_id": pid,
                    }
                )
    return personas


# ============================================================
# Analysis Functions
# ============================================================


def categorize_change(change: float) -> str:
    """Categorize change into bins."""
    if change <= -1.0:
        return "large_decrease"
    elif change <= -0.5:
        return "medium_decrease"
    elif change < -0.1:
        return "small_decrease"
    elif change <= 0.1:
        return "stable"
    elif change < 0.5:
        return "small_increase"
    elif change < 1.0:
        return "medium_increase"
    else:
        return "large_increase"


def check_hypothesis(change: float, expected: str) -> str:
    """Check if change matches expected direction."""
    if expected == "+":
        return "match" if change > 0.1 else ("opposite" if change < -0.1 else "neutral")
    elif expected == "-":
        return "match" if change < -0.1 else ("opposite" if change > 0.1 else "neutral")
    else:  # "?"
        return "uncertain"


def analyze_event_results(results: List[Dict], event: str) -> Dict:
    """Analyze results for a single event."""
    traits = [
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
        "Neuroticism",
        "Openness",
    ]
    trait_abbr = {
        "Extraversion": "E",
        "Agreeableness": "A",
        "Conscientiousness": "C",
        "Neuroticism": "N",
        "Openness": "O",
    }

    expected = LIFE_EVENTS[event]["expected_changes"]

    analysis = {"event": event, "n_personas": len(results), "traits": {}}

    for trait in traits:
        changes = [r["change"][trait] for r in results]
        abbr = trait_abbr[trait]
        expected_dir = expected.get(abbr, "?")

        # Hypothesis checking
        hypothesis_results = [check_hypothesis(c, expected_dir) for c in changes]
        match_count = hypothesis_results.count("match")
        opposite_count = hypothesis_results.count("opposite")
        neutral_count = hypothesis_results.count("neutral")

        # Statistical test
        t_stat, p_value = stats.ttest_1samp(changes, 0)

        analysis["traits"][trait] = {
            "expected": expected_dir,
            "mean_change": np.mean(changes),
            "std": np.std(changes),
            "median": np.median(changes),
            "t_stat": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "hypothesis_match": match_count,
            "hypothesis_opposite": opposite_count,
            "hypothesis_neutral": neutral_count,
            "match_rate": match_count / len(changes) if expected_dir != "?" else None,
        }

    return analysis


# ============================================================
# Main Experiment Runner
# ============================================================


def run_full_experiment(
    model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
    method: str = "single",
    events: List[str] = None,
    personas: List[Dict] = None,
    output_dir: str = None,
    resume_from: str = None,
):
    """Run the full V2 experiment."""

    if method not in {"single", "multi"}:
        raise ValueError(f"Invalid method: {method}. Expected 'single' or 'multi'.")

    if events is None:
        events = list(LIFE_EVENTS.keys())

    if personas is None:
        personas = generate_all_personas()

    print(f"\n{'=' * 80}")
    print("FULL V2 EXPERIMENT: Personality Change After Life Events")
    print(f"{'=' * 80}")
    print(f"Model: {model}")
    print(f"Method: {method}")
    print(f"Personas: {len(personas)}")
    print(f"Events: {len(events)}")
    print(f"Total experiments: {len(personas) * len(events)}")
    print(f"{'=' * 80}\n")

    client = get_client()

    # Setup output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "results" / f"v2_full_{timestamp}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results if resuming
    existing_results = {}
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            with open(resume_path, "r") as f:
                data = json.load(f)
                for r in data.get("results", []):
                    key = f"{r['persona_id']}_{r['event']}"
                    existing_results[key] = r
            print(f"Loaded {len(existing_results)} existing results from {resume_from}")

    all_results = list(existing_results.values())

    # Measure baselines for all personas first
    print("\n--- Phase 1: Measuring Baselines ---\n")
    baselines = {}

    for i, persona in enumerate(personas):
        persona_id = persona["id"]

        # Check if we have baseline from existing results
        baseline_key = f"{persona_id}_baseline"
        if any(r.get("persona_id") == persona_id for r in all_results):
            # Extract baseline from existing result
            for r in all_results:
                if r.get("persona_id") == persona_id:
                    existing_baseline = r["baseline"]
                    # Handle both old format (just trait_scores) and new format (dict)
                    if (
                        isinstance(existing_baseline, dict)
                        and "trait_scores" in existing_baseline
                    ):
                        baselines[persona_id] = existing_baseline
                    else:
                        # Old format - convert to new format (but we don't have raw logs)
                        baselines[persona_id] = {
                            "trait_scores": existing_baseline,
                            "item_scores": None,
                            "item_conversations": None,
                            "items_extracted": None,
                        }
                    break
            print(f"[{i + 1}/{len(personas)}] {persona_id}: Using existing baseline")
            continue

        print(
            f"[{i + 1}/{len(personas)}] {persona_id}: Measuring baseline...",
            end=" ",
            flush=True,
        )

        system_prompt = create_persona_system_prompt(
            persona["gender"], persona["continent"], persona["personality_id"]
        )

        try:
            if method == "single":
                baseline = measure_bfi_single_call(client, system_prompt, model)
                baselines[persona_id] = {
                    "method": baseline.get("method", "single"),
                    "trait_scores": baseline["trait_scores"],
                    "item_scores": baseline.get("item_scores"),
                    "raw_response": baseline.get("raw_response"),
                    "messages": baseline.get("messages"),
                    "items_extracted": baseline["items_extracted"],
                }
            else:
                baseline = measure_bfi_multi_call(client, system_prompt, model)
                baselines[persona_id] = {
                    "method": baseline.get("method", "multi"),
                    "trait_scores": baseline["trait_scores"],
                    "item_scores": baseline.get("item_scores"),
                    "item_conversations": baseline.get("item_conversations"),
                    "items_extracted": baseline["items_extracted"],
                }
            print(f"OK ({baseline['items_extracted']}/44 items)")
        except Exception as e:
            print(f"FAILED: {e}")
            baselines[persona_id] = None

        time.sleep(0.1)

    # Save baselines
    baselines_file = output_dir / "baselines.json"
    with open(baselines_file, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"\nBaselines saved to: {baselines_file}")

    # Run experiments for each event
    print("\n--- Phase 2: Running Event Experiments ---\n")

    total_experiments = len(personas) * len(events)
    completed = len(all_results)

    for event in events:
        event_info = LIFE_EVENTS[event]
        print(f"\n{'=' * 60}")
        print(f"Event: {event.upper()} ({event_info['domain']})")
        print(f"Expected changes: {event_info['expected_changes']}")
        print(f"{'=' * 60}")

        event_results = []

        for persona in personas:
            persona_id = persona["id"]
            result_key = f"{persona_id}_{event}"

            # Skip if already done
            if result_key in existing_results:
                event_results.append(existing_results[result_key])
                continue

            # Skip if no baseline
            if baselines.get(persona_id) is None:
                continue

            completed += 1
            print(
                f"[{completed}/{total_experiments}] {persona_id} × {event}...",
                end=" ",
                flush=True,
            )

            system_prompt = create_persona_system_prompt(
                persona["gender"], persona["continent"], persona["personality_id"]
            )

            try:
                # Event notification + reflection
                event_messages = [
                    {
                        "role": "user",
                        "content": event_info["notification"]
                        + "\n\n"
                        + event_info["reflection_prompt"],
                    }
                ]

                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(event_messages)
                reflection = call_llm(client, messages, model, max_tokens=500)

                event_messages.append({"role": "assistant", "content": reflection})

                # Post-event measurement
                if method == "single":
                    post_event = measure_bfi_single_call(
                        client, system_prompt, model, event_messages
                    )
                else:
                    post_event = measure_bfi_multi_call(
                        client, system_prompt, model, event_messages
                    )

                # Calculate changes
                baseline_trait_scores = baselines[persona_id]["trait_scores"]
                change = {}
                for trait in baseline_trait_scores:
                    change[trait] = (
                        post_event["trait_scores"][trait] - baseline_trait_scores[trait]
                    )

                result = {
                    "persona_id": persona_id,
                    "persona": persona,
                    "event": event,
                    "baseline": baselines[persona_id],
                    "post_event": {
                        "method": post_event.get("method", method),
                        "trait_scores": post_event["trait_scores"],
                        "item_scores": post_event.get("item_scores"),
                        "raw_response": post_event.get("raw_response"),
                        "messages": post_event.get("messages"),
                        "item_conversations": post_event.get("item_conversations"),
                        "items_extracted": post_event["items_extracted"],
                    },
                    "change": change,
                    "reflection": reflection,
                    "reflection_messages": messages
                    + [{"role": "assistant", "content": reflection}],
                }

                all_results.append(result)
                event_results.append(result)

                # Print changes
                changes_str = " ".join([f"{t[0]}:{change[t]:+.2f}" for t in change])
                print(f"OK | {changes_str}")

            except Exception as e:
                print(f"FAILED: {e}")

            time.sleep(0.1)

        # Analyze this event
        if event_results:
            event_analysis = analyze_event_results(event_results, event)

            # Print summary
            print(f"\n--- {event} Summary ({len(event_results)} personas) ---")
            print(
                f"{'Trait':<15} {'Expected':>10} {'Mean':>10} {'p-value':>10} {'Match%':>10}"
            )
            print("-" * 55)
            for trait, data in event_analysis["traits"].items():
                exp = data["expected"]
                mean = data["mean_change"]
                p = data["p_value"]
                match = data["match_rate"]
                match_str = f"{match * 100:.1f}%" if match is not None else "N/A"
                sig = "*" if data["significant"] else ""
                print(
                    f"{trait:<15} {exp:>10} {mean:>+10.3f} {p:>10.4f}{sig} {match_str:>10}"
                )

    # Save all results
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    results_file = output_dir / "full_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            convert_to_serializable(
                {
                    "config": {
                        "model": model,
                        "method": method,
                        "n_personas": len(personas),
                        "n_events": len(events),
                        "timestamp": datetime.now().isoformat(),
                    },
                    "results": all_results,
                }
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Total results: {len(all_results)}")
    print(f"Results saved to: {results_file}")
    print(f"{'=' * 80}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full V2 experiment")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-235B-A22B-Instruct-2507",
        help="Model name/path",
    )
    parser.add_argument(
        "--events",
        type=str,
        nargs="+",
        default=None,
        help="Specific events to run (default: all 11)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from existing results file",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="single",
        choices=["single", "multi"],
        help="Measurement method: single (1 call for 44 items) or multi (44 calls)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot with only 10 personas (Asian Male) and 3 events",
    )

    args = parser.parse_args()

    if args.pilot:
        # Pilot mode: 10 personas × 3 events = 30 experiments
        personas = [
            {
                "id": f"M_Asia_{pid}",
                "gender": "Male",
                "continent": "Asia",
                "personality_id": pid,
            }
            for pid in PERSONALITY_DESCRIPTIONS.keys()
        ]
        events = ["graduation", "promotion", "chronic_illness"]
        print("Running PILOT experiment (10 personas × 3 events)")
    else:
        personas = None  # Will generate all 100
        events = args.events  # Will use all 11 if None

    run_full_experiment(
        model=args.model,
        method=args.method,
        events=events,
        personas=personas,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )
