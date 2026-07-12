from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from colbert import Searcher
from colbert.infra import Run, RunConfig


class ColBERTRetriever:
    """
    Wrapper around ColBERT.

    The rest of the project should only use this class.
    """

    def __init__(
        self,
        project_root: str | Path,
        experiment: str = "phase3_colbert",
        index_name: str = "ansible_docs",
    ):

        self.project_root = Path(project_root)

        self.index_root = self.project_root / "colbert_data"

        self.doc_map_path = (
            self.project_root
            / "colbert_data"
            / "data"
            / "doc_map.json"
        )

        with open(self.doc_map_path) as f:
            self.doc_map = json.load(f)

        self.doc_by_pid = {
            int(doc["pid"]): doc
            for doc in self.doc_map
        }

        with Run().context(
            RunConfig(
                root=str(self.index_root),
                experiment=experiment,
            )
        ):
            self.searcher = Searcher(index=index_name)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        results = self.searcher.search(query, k=top_k)

        if len(results) == 3:
            pids, ranks, scores = results
        else:
            pids, scores = results
            ranks = range(1, len(pids) + 1)

        output = []

        for pid, rank, score in zip(pids, ranks, scores):

            if isinstance(pid, str):
                pid = int(pid)

            doc = self.doc_by_pid[int(pid)]

            output.append(
                {
                    "pid": pid,
                    "rank": int(rank),
                    "score": float(score),
                    "module": doc["module"],
                    "chunk": doc["chunk"],
                    "source": doc["source"],
                    "text": doc["text"],
                }
            )

        return output