from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VQASample:
    sample_id: str
    question: str
    images: list[str]
    gold_answers: list[str]
    metadata: dict = field(default_factory=dict)


def sample_to_dict(sample: VQASample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "images": list(sample.images),
        "gold_answers": list(sample.gold_answers),
        "metadata": dict(sample.metadata),
    }


def sample_from_dict(row: dict[str, Any]) -> VQASample:
    gold = row.get("gold_answers", row.get("answers", row.get("answer", [])))
    if isinstance(gold, str):
        gold_answers = [gold]
    elif isinstance(gold, (list, tuple)):
        gold_answers = [str(item) for item in gold if str(item).strip()]
    else:
        gold_answers = [str(gold)] if gold is not None else []
    images = row.get("images", row.get("image", []))
    if isinstance(images, str):
        image_list = [images]
    elif isinstance(images, (list, tuple)):
        image_list = [str(item) for item in images]
    elif images is None:
        image_list = []
    else:
        image_list = [str(images)]
    return VQASample(
        sample_id=str(row.get("sample_id", row.get("id", row.get("question_id", "")))),
        question=str(row.get("question", row.get("query", ""))).strip(),
        images=image_list,
        gold_answers=gold_answers,
        metadata=dict(row.get("metadata", {})),
    )


def toy_samples() -> list[VQASample]:
    return [
        VQASample(
            sample_id="toy_eiffel",
            question="Which city is the Eiffel Tower located in?",
            images=["image://eiffel"],
            gold_answers=["Paris"],
            metadata={"needs_search": False},
        ),
        VQASample(
            sample_id="toy_mona_lisa",
            question="Which museum displays the Mona Lisa?",
            images=["image://mona_lisa"],
            gold_answers=["Louvre Museum", "the Louvre"],
            metadata={"needs_search": True},
        ),
        VQASample(
            sample_id="toy_bridge",
            question="Which city is associated with the Golden Gate Bridge?",
            images=["image://golden_gate"],
            gold_answers=["San Francisco"],
            metadata={"needs_search": True},
        ),
        VQASample(
            sample_id="toy_liberty",
            question="Which city is the Statue of Liberty in?",
            images=["image://liberty"],
            gold_answers=["New York City", "New York"],
            metadata={"needs_search": True},
        ),
        VQASample(
            sample_id="toy_colosseum",
            question="Which city contains the Colosseum?",
            images=["image://colosseum"],
            gold_answers=["Rome"],
            metadata={"needs_search": False},
        ),
        VQASample(
            sample_id="toy_sydney",
            question="Which city is the Sydney Opera House in?",
            images=["image://opera_house"],
            gold_answers=["Sydney"],
            metadata={"needs_search": False},
        ),
        VQASample(
            sample_id="toy_taj",
            question="Which Indian city is home to the Taj Mahal?",
            images=["image://taj_mahal"],
            gold_answers=["Agra"],
            metadata={"needs_search": True},
        ),
        VQASample(
            sample_id="toy_giza",
            question="Near which city are the Pyramids of Giza?",
            images=["image://giza"],
            gold_answers=["Cairo"],
            metadata={"needs_search": True},
        ),
    ]
