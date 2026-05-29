from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VQASample:
    sample_id: str
    question: str
    images: list[str]
    gold_answers: list[str]
    metadata: dict = field(default_factory=dict)


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

