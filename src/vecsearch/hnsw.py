"""
Hierarchical Navigable Small World (HNSW) graph for approximate nearest
neighbor search (Malkov & Yashunin).

The structure
-------------
Every vector is a node. Each node gets a random top layer L drawn from an
exponentially decaying distribution, and exists in layers 0..L. Layer 0 holds
every node; each layer above holds roughly 1/M as many as the one below.
Within each layer a node keeps links to a bounded number of neighbors.

Search
------
Start at the entry node on the top layer. On each upper layer, greedily walk to
the neighbor closest to the query until nothing improves, then drop down a
layer using that node as the entry point. On layer 0, run a beam search that
keeps the best `ef` nodes seen (`ef` is the recall/speed knob).

Insert
------
A new node runs the same descent to find where it belongs. On each layer it
occupies, it beam-searches for candidates (`ef_construction` wide), picks up to
M diverse neighbors with a heuristic, links to them both ways, and trims any
neighbor whose link list overflowed.

Metric convention
-----------------
Your Metric returns SIMILARITY (higher = closer), not distance, so "better"
always means a larger number here, and heaps are arranged accordingly.
"""

import heapq
import math
import random

from vecsearch.metric import Metric


class HNSW:
    def __init__(
        self,
        data: dict[str, list[float]],
        metric: Metric,
        M: int = 16,
        ef_construction: int = 100,
        seed: int = 0,
        shuffle: bool = True,
        keep_pruned: bool = True,
    ) -> None:
        if M < 2:
            raise ValueError("M must be >= 2")

        self.data = data  # word -> unit vector. Shared, not copied.
        self.metric = metric
        self.M = M  # max links per node on layers >= 1, and links chosen at insert
        self.M0 = 2 * M  # max links per node on layer 0 (denser bottom layer)
        self.ef_construction = ef_construction
        self.keep_pruned = keep_pruned  # top up pruned neighbor lists (see _select)
        self.rng = random.Random(seed)  # seeded so builds are reproducible

        # Layer height distribution: level = floor(-ln(U) * mL). With mL = 1/ln(M)
        # each layer has about 1/M as many nodes as the one below it.
        self.level_mult = 1.0 / math.log(M)

        # Internal storage uses integer ids because they are cheaper to hash and
        # store than strings. Words map back only at the API boundary.
        self.words: list[str] = []
        self.vecs: list[list[float]] = []
        self.word_to_id: dict[str, int] = {}

        # links[node][layer] = list of neighbor ids. A node's top layer is
        # len(links[node]) - 1.
        self.links: list[list[list[int]]] = []

        self.entry = -1  # id of the entry node (a node on the top layer)
        self.max_level = -1  # top layer currently in the graph

        # Insert order affects graph quality. GloVe is sorted by word frequency,
        # so shuffling avoids feeding the graph a systematically ordered stream.
        order = list(data)
        if shuffle:
            self.rng.shuffle(order)
        for word in order:
            self._insert(word, data[word])

    # ------------------------------------------------------------------
    # CORE PRIMITIVE: BEAM SEARCH WITHIN ONE LAYER
    # ------------------------------------------------------------------

    def _search_layer(
        self,
        query: list[float],
        entries: list[tuple[float, int]],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """Beam search on one layer. Returns up to `ef` (similarity, id) pairs,
        best first.

        Two heaps drive it:
          candidates: nodes we still need to expand. Best similarity pops first,
                      so we store (-sim, id) in Python's min-heap.
          results:    the best `ef` nodes seen so far. Worst pops first (that
                      is the one we evict), so we store (sim, id) in a min-heap
                      and results[0] is always the current worst.
        """
        visited = {i for _, i in entries}  # never score a node twice
        candidates = [(-s, i) for s, i in entries]
        results = list(entries)
        heapq.heapify(candidates)
        heapq.heapify(results)

        while candidates:
            neg_sim, current = heapq.heappop(candidates)

            # Stop when the best unexpanded candidate is worse than everything
            # in a full results list: expanding it cannot improve the results.
            # (While results is not full, keep going.)
            if len(results) >= ef and -neg_sim < results[0][0]:
                break

            for neighbor in self.links[current][layer]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                sim = self.metric(query, self.vecs[neighbor])

                # Worth keeping if the beam has room or it beats the worst.
                if len(results) < ef or sim > results[0][0]:
                    heapq.heappush(candidates, (-sim, neighbor))
                    heapq.heappush(results, (sim, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)  # evict the worst

        return sorted(results, reverse=True)

    # ------------------------------------------------------------------
    # NEIGHBOR SELECTION HEURISTIC
    # ------------------------------------------------------------------

    def _select(self, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """Pick up to `m` diverse neighbors from `candidates`.

        `candidates` is (similarity to the reference node, id), sorted best
        first. The naive choice is the m closest, but those often sit in the
        same direction and all link to each other, leaving a cluster with no
        edge toward its surroundings. The heuristic accepts a candidate only if
        it is closer to the reference than to every neighbor already accepted,
        so each accepted neighbor "covers" a different direction.
        """
        selected: list[tuple[float, int]] = []
        skipped: list[tuple[float, int]] = []

        for sim_to_ref, cand in candidates:
            if len(selected) >= m:
                break
            cand_vec = self.vecs[cand]
            # Diverse enough if it is more similar to the reference than to any
            # already-selected neighbor. all() short-circuits on the first
            # violation, which saves distance computations.
            if all(
                sim_to_ref > self.metric(cand_vec, self.vecs[chosen])
                for _, chosen in selected
            ):
                selected.append((sim_to_ref, cand))
            else:
                skipped.append((sim_to_ref, cand))

        # Optionally top up with the best rejected candidates so nodes are not
        # left under-connected. Helps recall at small M, costs a little memory.
        if self.keep_pruned:
            for pair in skipped:
                if len(selected) >= m:
                    break
                selected.append(pair)

        return [i for _, i in selected]

    # ------------------------------------------------------------------
    # ADDING POINTS
    # ------------------------------------------------------------------

    def add(self, word: str, vector: list[float]) -> None:
        """Insert one new (already L2-normalized) vector into the index."""
        if word in self.data:
            raise ValueError(f"{word} already in index")
        self.data[word] = vector
        self._insert(word, vector)

    def _insert(self, word: str, vector: list[float]) -> None:
        node = len(self.words)

        # 1. Draw this node's top layer. 1 - random() lies in (0, 1], so the
        #    log never sees zero.
        level = int(-math.log(1.0 - self.rng.random()) * self.level_mult)

        self.words.append(word)
        self.vecs.append(vector)
        self.word_to_id[word] = node
        self.links.append([[] for _ in range(level + 1)])

        # 2. The very first node becomes the entry point, and nothing to link.
        if self.entry == -1:
            self.entry = node
            self.max_level = level
            return

        # 3. Descend from the top to just above this node's top layer, greedily
        #    (ef=1). We only need a good entry point, so no linking happens.
        ep = [(self.metric(vector, self.vecs[self.entry]), self.entry)]
        for layer in range(self.max_level, level, -1):
            ep = self._search_layer(vector, ep, 1, layer)

        # 4. On every layer this node occupies (and that already exists), find
        #    candidates with a wide beam, choose diverse neighbors and link.
        for layer in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(vector, ep, self.ef_construction, layer)
            m_max = self.M0 if layer == 0 else self.M

            neighbors = self._select(found, self.M)
            self.links[node][layer] = neighbors

            # Links are bidirectional. Adding the reverse edge can push a
            # neighbor past its cap, so re-select that neighbor's best links.
            for n in neighbors:
                n_links = self.links[n][layer]
                n_links.append(node)
                if len(n_links) > m_max:
                    scored = sorted(
                        ((self.metric(self.vecs[n], self.vecs[x]), x) for x in n_links),
                        reverse=True,
                    )
                    self.links[n][layer] = self._select(scored, m_max)

            # The candidates found here seed the search on the next layer down.
            ep = found

        # 5. A node taller than the current graph becomes the new entry point.
        if level > self.max_level:
            self.entry = node
            self.max_level = level

    # ------------------------------------------------------------------
    # SEARCHING
    # ------------------------------------------------------------------

    def search(
        self,
        query: list[float],
        k: int = 10,
        ef: int = 50,
        exclude: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return the approximate top-k (word, similarity) pairs for `query`.

        ef: beam width on layer 0. Bigger means better recall and more distance
        computations. It must be at least k, and one more if we exclude a word,
        otherwise the beam could not even hold k results.
        """
        if self.entry == -1:
            return []
        ef = max(ef, k + (1 if exclude is not None else 0))

        # Greedy descent through the upper layers (beam of 1).
        ep = [(self.metric(query, self.vecs[self.entry]), self.entry)]
        for layer in range(self.max_level, 0, -1):
            ep = self._search_layer(query, ep, 1, layer)

        # Wide beam search on the full layer 0 graph.
        found = self._search_layer(query, ep, ef, 0)

        out: list[tuple[str, float]] = []
        for sim, i in found:
            word = self.words[i]
            if word == exclude:
                continue
            out.append((word, sim))
            if len(out) == k:
                break
        return out

    def find_closest(
        self, target_word: str, k: int = 5, ef: int = 50
    ) -> list[tuple[str, float]]:
        """Same interface as KNN.find_closest, so the bench harness works."""
        if target_word not in self.data:
            raise ValueError(f"{target_word} not in vocabulary")
        return self.search(self.data[target_word], k, ef, exclude=target_word)
