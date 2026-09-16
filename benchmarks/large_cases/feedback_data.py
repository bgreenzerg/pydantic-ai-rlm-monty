"""Reproducible, fictitious customer-feedback corpus for the live benchmark."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DATASET_SIZE = 5_000
RANDOM_SEED = 20260911


def _candidate_sets() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Choose a reproducible, non-adjacent target sample from scattered candidate IDs."""
    rng = random.Random(RANDOM_SEED + 41)
    candidates = tuple(sorted(rng.sample(range(1, DATASET_SIZE + 1), 120)))
    while True:
        cohort = tuple(sorted(rng.sample(candidates, 24)))
        positions = sorted(candidates.index(index) for index in cohort)
        if all(right - left > 1 for left, right in itertools.pairwise(positions)):
            return candidates, cohort


RECENT_CANDIDATE_INDICES, COHORT_INDICES = _candidate_sets()
SAME_METADATA_DECOY_INDICES = tuple(index for index in RECENT_CANDIDATE_INDICES if index not in COHORT_INDICES)
SUPPORTING_COUNT = len(COHORT_INDICES)
REPRESENTATIVE_IDS = tuple(f"FB-2026-{index:04d}" for index in COHORT_INDICES[:3])
DATASET_PATH = Path(__file__).resolve().parent / "data" / "customer_feedback.jsonl"
EXPECTED_SHA256 = "aebfea7895c38ecace581a7feb69ee2b7e59677aa5df7db8645add9d1d75e915"

TASK = (
    "Investigate recent one- and two-star customer feedback. First use the structured feedback fields to identify "
    "concentrated recent low-rating segments, then determine whether there is a coherent checkout defect rather than "
    "unrelated dissatisfaction. Group reports by semantic equivalence of the failure mechanism, not merely keyword "
    "overlap. Before drawing conclusions, obtain an independent focused review of the narrowed candidate comments; "
    "then describe the affected pattern, estimate its supporting count, recommend a remediation, and cite "
    "representative feedback IDs."
)

_LOCALES = ("da-DK", "en-GB", "de-DE", "sv-SE", "nl-NL")
_CHANNELS = ("web", "mobile_web", "app", "email_survey")
_DEVICES = ("desktop_chrome", "mobile_safari", "mobile_chrome", "ios_app", "android_app")
_STAGES = ("browse", "product", "basket", "checkout", "delivery", "returns")
_POSITIVE = (
    "Quick delivery and the quality matched the photos.",
    "Checkout was smooth and the order confirmation arrived immediately.",
    "The address suggestions saved time and selected my street correctly.",
    "Helpful support and a simple return process.",
    "Good range of products and clear stock information.",
)
_DECOYS = (
    "The parcel arrived late and I had no useful tracking update.",
    "The colour looked different from the product photos.",
    "A discount code disappeared before payment and support could not restore it.",
    "My card was declined twice even though it works elsewhere.",
    "The return label took several days to arrive.",
    "The size guide was inaccurate, so I need an exchange.",
    "The delivery point changed after dispatch without a clear explanation.",
    "I could not sign in after resetting my password.",
)
_COHORT_COMMENTS = (
    "The order summary changed 2. th. to just the street and number after I chose a proposed location.",
    "My complete delivery line was visible until the final address match replaced it with the building only.",
    "Tapping the recommended road caused the part identifying my flat to disappear before payment.",
    "The confirmation retained 18B but not the letter that tells the courier which door to use.",
    "I selected a result from the list and the floor detail I had typed was no longer on the receipt.",
    "The phone checkout turned a specific home address into the shared entrance without asking.",
    "After the lookup accepted my street, the extra residence detail was quietly removed from the order.",
    "Everything was correct while entering it; choosing a suggestion left only the main house reference.",
    "The delivery destination lost its apartment part at the moment the suggested address was applied.",
    "An automatic match overwrote the last line of my address, which was needed to find my unit.",
    "I chose the correct street recommendation, then the stairwell notation vanished from checkout.",
    "The cart had my flat code, but the order placed after accepting the address result did not.",
    "Selecting the proposed address reduced my destination to a place where several households share one door.",
    "My order went through with the road intact but without the sub-address I supplied before choosing the match.",
    "The address helper kept the house number and silently discarded the apartment suffix on the final screen.",
    "I used the suggested destination and the courier-facing detail after the comma disappeared.",
    "The checkout recommendation converted my full address into one missing the floor and side information.",
    "Before I tapped the option, the delivery field included my unit; afterwards it no longer did.",
    "The selected result replaced a detailed residence with only the block address when I confirmed the basket.",
    "I recognised the street in the suggestion, but it erased the part of the address that distinguishes my flat.",
    "A matching address appeared, and accepting it removed the door reference needed inside the building.",
    "The final order omitted my apartment marker even though the address form had accepted it earlier.",
    "Choosing from autocomplete caused the ending of my delivery address to be lost without a warning.",
    "The suggested street was right, but confirmation dropped the unit-level information from the shipment address.",
)
_SAME_METADATA_DECOY_MECHANISMS = (
    "The address suggestion chose a similarly named road in another town, while my apartment detail remained visible.",
    "No suggestion appeared for our new street, so I could not complete checkout.",
    "My door number was correct on the receipt, but the parcel arrived two days late.",
    "The delivery address was intact, yet the order was routed to the wrong pickup point.",
    "Autocomplete proposed my old address and editing it before payment took too many taps.",
    "The map pin behind the suggested address was inaccurate although all the typed fields were preserved.",
    "The address list did not include the building, so I had to abandon the order rather than select a match.",
    "My unit number stayed in place, but the payment button froze after the address form validated it.",
    "Checkout asked for a door code too late, without changing any part of the delivery address.",
    "The recommended street name was misspelled, but the confirmation still contained my complete address.",
    "Selecting an address switched delivery from home service to a locker with no warning.",
    "A valid postal code was rejected despite the correct street and door number.",
)
_DECOY_CONTEXTS = (
    "This happened while ordering a birthday gift.",
    "I was using the site during a lunch break.",
    "Support's first reply did not address the inconvenience.",
    "The problem occurred after I had already checked the basket twice.",
    "I noticed it before placing the order and had to start over.",
    "It made a routine purchase take much longer than expected.",
    "I compared the details with a previous successful order.",
    "The experience was frustrating even though the address text itself stayed available.",
)
_COHORT_POSITIONS = {index: position for position, index in enumerate(COHORT_INDICES)}
_DECOY_POSITIONS = {index: position for position, index in enumerate(SAME_METADATA_DECOY_INDICES)}
_CANDIDATE_POSITIONS = {index: position for position, index in enumerate(RECENT_CANDIDATE_INDICES)}


def feedback_id(index: int) -> str:
    """Return the stable public identifier for a one-based record index."""
    return f"FB-2026-{index:04d}"


def _ordinary_record(index: int, rng: random.Random) -> dict[str, Any]:
    rating = 5 if index % 10 < 5 else 4 if index % 10 < 7 else 3 if index % 10 < 8 else rng.choice((1, 2))
    base_comment = rng.choice(_POSITIVE if rating >= 4 else _DECOYS)
    context = rng.choice(("for a household order", "while shopping from home", "on a weekday evening", "for a gift"))
    comment = f"{base_comment} This was {context}."
    return {
        "feedback_id": feedback_id(index),
        "timestamp": (datetime(2025, 10, 1, tzinfo=UTC) + timedelta(days=(index * 17) % 320, hours=index % 24)).isoformat(),
        "rating": rating,
        "locale": rng.choice(_LOCALES),
        "channel": rng.choice(_CHANNELS),
        "device": rng.choice(_DEVICES),
        "journey_stage": rng.choice(_STAGES),
        "verified_purchase": bool(index % 3),
        "comment": comment,
    }


def _cohort_record(index: int) -> dict[str, Any]:
    """Return one seeded, low-rating symptom report without an issue label."""
    offset = _COHORT_POSITIONS[index]
    return {
        "feedback_id": feedback_id(index),
        "timestamp": _candidate_timestamp(index).isoformat(),
        "rating": 1 if offset % 3 else 2,
        "locale": "da-DK",
        "channel": "mobile_web",
        "device": "mobile_safari",
        "journey_stage": "checkout",
        "verified_purchase": True,
        "comment": _COHORT_COMMENTS[offset],
    }


def _same_metadata_decoy_record(index: int) -> dict[str, Any]:
    """Return a recent low-rating checkout complaint with matching metadata but a different mechanism."""
    offset = _DECOY_POSITIONS[index]
    return {
        "feedback_id": feedback_id(index),
        "timestamp": _candidate_timestamp(index).isoformat(),
        "rating": 1 if offset % 4 else 2,
        "locale": "da-DK",
        "channel": "mobile_web",
        "device": "mobile_safari",
        "journey_stage": "checkout",
        "verified_purchase": bool(offset % 2),
        "comment": (
            f"{_SAME_METADATA_DECOY_MECHANISMS[offset % len(_SAME_METADATA_DECOY_MECHANISMS)]} "
            f"{_DECOY_CONTEXTS[offset // len(_SAME_METADATA_DECOY_MECHANISMS)]}"
        ),
    }


def _candidate_timestamp(index: int) -> datetime:
    """Give every same-metadata candidate a unique, shuffled recent timestamp."""
    position = _CANDIDATE_POSITIONS[index]
    shuffled_hours = (position * 47) % len(RECENT_CANDIDATE_INDICES) * 3
    return datetime(2026, 8, 1, 8, tzinfo=UTC) + timedelta(hours=shuffled_hours)


def records() -> list[dict[str, Any]]:
    """Create exactly ``DATASET_SIZE`` deterministic feedback records."""
    rng = random.Random(RANDOM_SEED)
    return [
        _cohort_record(index)
        if index in COHORT_INDICES
        else _same_metadata_decoy_record(index)
        if index in SAME_METADATA_DECOY_INDICES
        else _ordinary_record(index, rng)
        for index in range(1, 5_001)
    ]


def ensure_dataset(path: Path = DATASET_PATH) -> Path:
    """Write the deterministic JSONL corpus only when it is absent or stale."""
    expected = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records())
    if not path.exists() or path.read_text(encoding="utf-8") != expected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8", newline="")
    return path


def load_dataset(path: Path = DATASET_PATH) -> str:
    """Ensure and return the JSONL text supplied as opaque RLM context."""
    content = ensure_dataset(path).read_text(encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"feedback fixture digest mismatch: {digest}")
    return content
