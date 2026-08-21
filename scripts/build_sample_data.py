"""Generate a small but consistent HotpotQA-style dataset for offline testing.

The generated data follows the official HotpotQA schema so the same loaders
work for both real and synthetic data.  Every question is grounded in
``FACTS`` below so the synthetic passages never contradict themselves.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Knowledge base – triples we know are true.
# ---------------------------------------------------------------------------
FACTS: List[Tuple[str, str, str]] = [
    ("Inception",       "director",   "Christopher Nolan"),
    ("The Matrix",      "director",   "Lana Wachowski"),
    ("Interstellar",    "director",   "Christopher Nolan"),
    ("The Dark Knight", "director",   "Christopher Nolan"),
    ("Pulp Fiction",    "director",   "Quentin Tarantino"),
    ("Forrest Gump",    "director",   "Robert Zemeckis"),
    ("Inception",       "starring",   "Leonardo DiCaprio"),
    ("The Matrix",      "starring",   "Keanu Reeves"),
    ("Interstellar",    "starring",   "Matthew McConaughey"),
    ("Pulp Fiction",    "starring",   "John Travolta"),
    ("Forrest Gump",    "starring",   "Tom Hanks"),
    ("Frankenstein",    "author",     "Mary Shelley"),
    ("Dracula",         "author",     "Bram Stoker"),
    ("The Shining",     "author",     "Stephen King"),
    ("It",              "author",     "Stephen King"),
    ("Christopher Nolan", "nationality", "United Kingdom"),
    ("Lana Wachowski",    "nationality", "United States"),
    ("Quentin Tarantino", "nationality", "United States"),
    ("Robert Zemeckis",   "nationality", "United States"),
    ("Leonardo DiCaprio", "nationality", "United States"),
    ("Mary Shelley",      "nationality", "United Kingdom"),
    ("Bram Stoker",       "nationality", "United Kingdom"),
    ("Stephen King",      "nationality", "United States"),
    ("Tom Hanks",         "nationality", "United States"),
]

VERB = {
    "director":   "was directed by",
    "starring":   "stars",
    "author":     "was written by",
    "nationality": "is from",
}


def _passage(title: str, relation: str, obj: str) -> List[str]:
    v = VERB[relation]
    return [
        f"{title} is a well-known work.",
        f"{title} {v} {obj}.",
        f"This fact is frequently mentioned in reference works about {obj}.",
    ]


def _facts_about(entity: str) -> List[Tuple[str, str, str]]:
    return [f for f in FACTS if f[0] == entity]


# ---------------------------------------------------------------------------
# Question generators
# ---------------------------------------------------------------------------
def _bridge_director_nationality(rng: random.Random) -> Dict[str, Any]:
    """Q: who directed X and what country is that director from?"""
    movie = rng.choice([f[0] for f in FACTS if f[1] == "director"])
    director = next(o for s, r, o in FACTS if s == movie and r == "director")
    nationality = next((o for s, r, o in FACTS if s == director and r == "nationality"),
                        "Unknown")
    question = (f"Which director directed {movie}, and what country is that "
                 f"director from?")
    return {
        "_id": f"bridge_dir_nat_{movie}".replace(" ", "_"),
        "question": question,
        "answer": nationality,
        "supporting_facts": [
            {"title": f"Article about {movie}", "sent_id": 1},
            {"title": f"Biography of {director}", "sent_id": 1},
        ],
        "context": [
            [f"Article about {movie}",
             _passage(movie, "director", director) +
             [f"{movie} is considered influential in its genre."]],
            [f"Biography of {director}",
             [f"{director} is a prominent filmmaker.",
              f"{director} {VERB['nationality']} {nationality}.",
              f"{director} studied at university."]],
            ["Other notable works",
             ["Many other notable works exist in the same period.",
              "Some of them are mentioned in encyclopedias."]],
        ],
        "type": "bridge", "level": "hard",
    }


def _bridge_author_nationality(rng: random.Random) -> Dict[str, Any]:
    """Q: the author of X was from which country?"""
    book = rng.choice([f[0] for f in FACTS if f[1] == "author"])
    author = next(o for s, r, o in FACTS if s == book and r == "author")
    nationality = next((o for s, r, o in FACTS if s == author and r == "nationality"),
                        "Unknown")
    question = f"The author of {book} was from which country?"
    return {
        "_id": f"bridge_author_nat_{book}".replace(" ", "_"),
        "question": question,
        "answer": nationality,
        "supporting_facts": [
            {"title": f"Article about {book}", "sent_id": 1},
            {"title": f"Biography of {author}", "sent_id": 1},
        ],
        "context": [
            [f"Article about {book}",
             _passage(book, "author", author) +
             [f"{book} has been reprinted many times."]],
            [f"Biography of {author}",
             [f"{author} is a celebrated author.",
              f"{author} {VERB['nationality']} {nationality}.",
              f"{author} wrote several books."]],
            ["Other authors",
             ["Many other authors have written similar works.",
              "Their works are also widely studied."]],
        ],
        "type": "bridge", "level": "medium",
    }


def _comparison_directors(rng: random.Random) -> Dict[str, Any]:
    """Q: who directed A and B?"""
    movies = rng.sample([f[0] for f in FACTS if f[1] == "director"], 2)
    directors = [next(o for s, r, o in FACTS if s == m and r == "director")
                  for m in movies]
    question = f"Which director directed both {movies[0]} and {movies[1]}?"
    # If they share a director, answer that director; otherwise ask for both.
    if directors[0] == directors[1]:
        answer = directors[0]
    else:
        answer = f"{directors[0]} and {directors[1]}"
    return {
        "_id": f"comparison_dirs_{'_'.join(movies)}".replace(" ", "_"),
        "question": question,
        "answer": answer,
        "supporting_facts": [
            {"title": f"Article about {movies[0]}", "sent_id": 1},
            {"title": f"Article about {movies[1]}", "sent_id": 1},
        ],
        "context": [
            [f"Article about {movies[0]}",
             _passage(movies[0], "director", directors[0]) +
             [f"{movies[0]} was released to critical acclaim."]],
            [f"Article about {movies[1]}",
             _passage(movies[1], "director", directors[1]) +
             [f"{movies[1]} was released to critical acclaim."]],
            ["Other films",
             ["Several other films were released in the same period."]],
        ],
        "type": "comparison", "level": "easy",
    }


def _actor_nationality(rng: random.Random) -> Dict[str, Any]:
    """Q: which country is actor X from?"""
    actors = [f[0] for f in FACTS if f[1] == "nationality" and
                f[0] in {a for s, r, a in FACTS if r == "starring"}]
    actor = rng.choice(actors)
    nationality = next(o for s, r, o in FACTS if s == actor and r == "nationality")
    question = f"Which country is the actor {actor} from?"
    return {
        "_id": f"actor_nat_{actor}".replace(" ", "_"),
        "question": question,
        "answer": nationality,
        "supporting_facts": [
            {"title": f"Biography of {actor}", "sent_id": 1},
        ],
        "context": [
            [f"Biography of {actor}",
             [f"{actor} is a well-known actor.",
              f"{actor} {VERB['nationality']} {nationality}.",
              f"{actor} has starred in many films."]],
            ["Other actors",
             ["Many other actors are well known internationally."]],
        ],
        "type": "bridge", "level": "easy",
    }


def build(n: int, seed: int = 0) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    generators = [
        _bridge_director_nationality,
        _bridge_author_nationality,
        _comparison_directors,
        _actor_nationality,
    ]
    out: List[Dict[str, Any]] = []
    for i in range(n):
        gen = generators[i % len(generators)]
        out.append(gen(rng))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sample_hotpotqa.json")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    records = build(args.n, seed=args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} synthetic HotpotQA-style records to {args.output}")


if __name__ == "__main__":
    main()
