"""
Hierarchical Navigable Small World (HNSW) over a contiguous numpy vector store.

This is the same algorithm as `hnsw.py` (Malkov & Yashunin) with exactly one
structural change: vectors live in a single `(N, D)` array and every batch of
similarities is one matrix-vector product instead of a Python loop of pairwise
dots. The graph logic -- layered random levels, beam search per layer, the
diversity heuristic, bidirectional self-trimming links -- is unchanged, so the
two indexes answer the same queries with the same knobs (`M`, `ef`,
`ef_construction`, `ef`).

Where the speedup actually comes from
------------------------------------
Measured on this project's data shape (d=100, 32 neighbors scored per pop):

    per-pair math.sumprod      20.2 us/pop
    numpy float32 matvec        2.5 us/pop     <- what this module does
    numpy float64 matvec        7.8 us/pop

The beam loop itself is unchanged and still runs one expansion at a time, since
each pop decides which node to expand next. Only the scoring of that pop's
neighbors is batched.

What is deliberately NOT vectorized
-----------------------------------
`_select` stays pure Python. It looks vectorizable and is not: the diversity
heuristic accepts a candidate only if it beats every already-accepted neighbor,
and `all()` short-circuits, so it runs on average one or two comparisons per
candidate. Measured at d=100, a precomputed 100x100 Gram matrix was 2.2x
SLOWER than the per-pair loop, and batching the inner comparison across the
accepted set was 2.6x slower -- numpy's per-call overhead dominates when the
inner loop only does one or two 100-element dots. `math.sumprod` over plain
Python lists has almost no call overhead, so it stays.

Assumption: inner product on unit vectors
-----------------------------------------
`hnsw.py` takes an arbitrary `Metric` callable and routes every distance through
it. That is not possible here: an arbitrary Python callable cannot be batched
into one matmul, which is the entire point of this module. Scoring is therefore
a raw dot product, which equals cosine similarity exactly when both vectors are
L2-normalized -- which `load_normalized_data` guarantees. This is why the
constructor takes no `metric` argument.

Counting
--------
A matvec computes many distances in a single call, so the injected
`CountingMetric` used by the benchmark cannot see this module's work. Following
the `RPForest.plane_evals` precedent, the index tallies its own
`dist_evals` (counted in vectors scored, not matvec calls) and the benchmark
adds it into the per-query distance total.
"""

import heapq
import math
import random

import numpy as np


class HNSWNP:
    """HNSW with a contiguous numpy vector store and batched scoring.

    dtype selects the vector store precision. float32 is the default because it
    is ~3x faster per matvec and ~2.5x smaller in memory, but it is not a pure
    speed change: the reduced precision flips near-tie comparisons in the
    `_select` diversity heuristic, and measured over synthetic data roughly 10 of
    16 links per node come out different. float32 therefore builds a genuinely
    different graph, not a rounding of the same one. Benchmark both dtypes
    before trusting either.
    """

    MIN_CAPACITY = 1024

    def __init__(
        self,
        data: dict[str, list[float]],
        M: int = 16,
        ef_construction: int = 100,
        seed: int = 0,
        shuffle: bool = True,
        keep_pruned: bool = True,
        dtype: type = np.float32,
    ) -> None:
        if M < 2:
            raise ValueError("M must be >= 2")

        self.M = M  # max links per node on layers >= 1, and links chosen at insert
        self.M0 = 2 * M  # max links per node on layer 0 (denser bottom layer)
        self.ef_construction = ef_construction
        self.keep_pruned = keep_pruned  # top up pruned neighbor lists (see _select)
        self.dtype = np.dtype(dtype)
        self.rng = random.Random(seed)  # seeded so builds are reproducible

        # Layer height distribution: level = floor(-ln(U) * mL). With mL = 1/ln(M)
        # each layer has about 1/M as many nodes as the one below it.
        self.level_mult = 1.0 / math.log(M)

        # Real work this index does that CountingMetric can't see, since none of
        # it goes through an injected metric. The benchmark adds this to the
        # distance count, exactly as it does RPForest.plane_evals.
        self.dist_evals = 0

        # Internal storage uses integer ids because they are cheaper to hash and
        # store than strings. Words map back only at the API boundary.
        self.words: list[str] = []
        self.word_to_id: dict[str, int] = {}

        # The vector store. Grown by doubling so add() stays amortized O(D).
        words = list(data)
        self.dim = len(next(iter(data.values()))) if words else 0
        self.matrix = np.empty(
            (max(len(words), self.MIN_CAPACITY), self.dim), dtype=self.dtype
        )
        # `_select` reads Python lists, not the matrix: see the module docstring.
        self.vecs: list[list[float]] = [data[w] for w in words]
        for i, word in enumerate(words):
            self.matrix[i] = data[word]
            self.word_to_id[word] = i
        self.words = words
        self.n = len(words)

        # links[node][layer] is a fixed-width int32 buffer of neighbor ids and
        # nlinks[node][layer] is how many of those slots are live. Storing ids as
        # an array rather than a Python list lets the beam search index the
        # matrix with a view and skip a list->array conversion on every pop
        # (measured 0.9 us/pop, 36% of the remaining scoring time). It also drops
        # adjacency memory ~10x. The width is the cap that layer enforces.
        # A node's top layer is len(links[node]) - 1.
        #
        # These are indexed BY NODE ID, not by insertion order. The slots are
        # filled in as nodes are inserted, and insert order is shuffled, so a
        # plain append would put a node at the wrong offset.
        self.links: list[list[np.ndarray] | None] = [None] * self.n
        self.nlinks: list[list[int] | None] = [None] * self.n

        # Reused across scoring calls to avoid allocating a fresh (k, D) block
        # on every pop. Only valid until the next _score call, which is fine
        # because every caller consumes the result immediately.
        self._scratch = np.empty((0, self.dim), dtype=self.dtype)

        self.entry = -1  # id of the entry node (a node on the top layer)
        self.max_level = -1  # top layer currently in the graph

        # Insert order affects graph quality. GloVe is sorted by word frequency,
        # so shuffling avoids feeding the graph a systematically ordered stream.
        order = list(range(self.n))
        if shuffle:
            self.rng.shuffle(order)
        for i in order:
            self._insert(i)

    # ------------------------------------------------------------------
    # VECTOR STORE
    # ------------------------------------------------------------------

    def _score(self, ids: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Dot `query` against every id in `ids`, as one matvec.

        `ids` is already restricted to unvisited nodes, so this is also the
        place the per-pop distance count is tallied.
        """
        k = ids.size
        if self._scratch.shape[0] < k:
            self._scratch = np.empty((k, self.dim), dtype=self.dtype)
        block = self._scratch[:k]
        np.take(self.matrix, ids, axis=0, out=block)
        self.dist_evals += k
        return block @ query

    def _push_vector(self, vector: list[float]) -> int:
        """Append one vector to the store, growing it by doubling. Returns its id."""
        if self.n == self.matrix.shape[0]:
            grown = np.empty(
                (max(2 * self.n, self.MIN_CAPACITY), self.dim), dtype=self.dtype
            )
            grown[: self.n] = self.matrix[: self.n]
            self.matrix = grown
        self.matrix[self.n] = vector
        self.vecs.append(vector)
        node = self.n
        self.n += 1
        return node

    # ------------------------------------------------------------------
    # LINK STORAGE
    # ------------------------------------------------------------------

    def _slot(self, layer: int) -> int:
        """Width of a link buffer on `layer`: M above layer 0, 2M on layer 0."""
        return self.M0 if layer == 0 else self.M

    def _slots(self, node: int) -> tuple[list[np.ndarray], list[int]]:
        """Link buffers and live counts for an already-inserted node.

        `links`/`nlinks` are pre-sized with None placeholders because insert
        order is shuffled, so a node's slot is filled when it is inserted. Every
        reader goes through here, which both narrows the type and turns
        "searched a node that does not exist" into a clear failure.
        """
        links = self.links[node]
        counts = self.nlinks[node]
        assert links is not None and counts is not None, f"node {node} not inserted"
        return links, counts

    def _neighbors(self, node: int, layer: int) -> np.ndarray:
        """Live neighbor ids of `node` on `layer`, as a view into its buffer."""
        links, counts = self._slots(node)
        return links[layer][: counts[layer]]

    def _set_links(self, node: int, layer: int, ids) -> None:
        """Overwrite `node`'s links on `layer`, truncating to the slot width."""
        links, counts = self._slots(node)
        buf = links[layer]
        chosen = list(ids)[: buf.size]
        if chosen:
            buf[: len(chosen)] = chosen
        counts[layer] = len(chosen)

    # ------------------------------------------------------------------
    # CORE PRIMITIVE: BEAM SEARCH WITHIN ONE LAYER
    # ------------------------------------------------------------------

    def _search_layer(
        self,
        query: np.ndarray,
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

        The loop is one expansion at a time -- each pop decides the next node to
        look at, so the traversal cannot be batched. What is batched is the
        scoring of that node's unvisited neighbors, which is the part that costs
        the wall clock. Admission stays sequential, so a node is still tested
        against the current worst of the beam one at a time.
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

            raw = self._neighbors(current, layer)
            if raw.size == 0:
                continue

            # Filter to unvisited, then score the survivors in one matvec.
            fresh = raw[[i not in visited for i in raw.tolist()]]
            if fresh.size == 0:
                continue
            sims = self._score(fresh, query).tolist()

            for neighbor, sim in zip(fresh.tolist(), sims):
                visited.add(neighbor)

                # Worth keeping if the beam has room or it beats the worst.
                if len(results) < ef or sim > results[0][0]:
                    heapq.heappush(candidates, (-sim, neighbor))
                    heapq.heappush(results, (sim, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)  # evict the worst

        return sorted(results, reverse=True)

    # ------------------------------------------------------------------
    # NEIGHBOR SELECTION HEURISTIC  (intentionally not vectorized)
    # ------------------------------------------------------------------

    def _select(self, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """Pick up to `m` diverse neighbors from `candidates`.

        `candidates` is (similarity to the reference node, id), sorted best
        first. The naive choice is the m closest, but those often sit in the
        same direction and all link to each other, leaving a cluster with no
        edge toward its surroundings. The heuristic accepts a candidate only if
        it is closer to the reference than to every neighbor already accepted,
        so each accepted neighbor "covers" a different direction.

        This deliberately stays on per-pair math.sumprod over Python lists. The
        all() below short-circuits, so the inner loop runs one or two
        comparisons per candidate, and at that size numpy's per-call overhead
        costs more than the arithmetic it replaces. See the module docstring for
        the numbers.
        """
        selected: list[tuple[float, int]] = []
        skipped: list[tuple[float, int]] = []

        for sim_to_ref, cand in candidates:
            if len(selected) >= m:
                break
            cand_vec = self.vecs[cand]
            # Diverse enough if it is more similar to the reference than to any
            # already-selected neighbor. The loop breaks on the first violation,
            # which saves distance computations, so the count here matches what
            # the pure-Python math.sumprod below actually evaluates.
            diverse = True
            for _, chosen in selected:
                self.dist_evals += 1
                if not sim_to_ref > math.sumprod(cand_vec, self.vecs[chosen]):
                    diverse = False
                    break
            if diverse:
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
        if word in self.word_to_id:
            raise ValueError(f"{word} already in index")
        self._insert(self._push_vector(vector), word=word)

    def _insert(self, node: int, word: str | None = None) -> None:
        if word is not None:
            self.words.append(word)
            self.word_to_id[word] = node

        # 1. Draw this node's top layer. 1 - random() lies in (0, 1], so the
        #    log never sees zero.
        level = int(-math.log(1.0 - self.rng.random()) * self.level_mult)

        if node == len(self.links):  # a node arriving via add()
            self.links.append(None)
            self.nlinks.append(None)
        self.links[node] = [
            np.empty(self._slot(layer), dtype=np.int32) for layer in range(level + 1)
        ]
        self.nlinks[node] = [0] * (level + 1)

        # 2. The very first node becomes the entry point, and nothing to link.
        if self.entry == -1:
            self.entry = node
            self.max_level = level
            return

        # A view of this node's row. Safe: the store only ever grows in _insert's
        # own _push_vector, which already happened.
        vector = self.matrix[node]

        # 3. Descend from the top to just above this node's top layer, greedily
        #    (ef=1). We only need a good entry point, so no linking happens.
        self.dist_evals += 1
        ep = [(float(vector @ self.matrix[self.entry]), self.entry)]
        for layer in range(self.max_level, level, -1):
            ep = self._search_layer(vector, ep, 1, layer)

        # 4. On every layer this node occupies (and that already exists), find
        #    candidates with a wide beam, choose diverse neighbors and link.
        for layer in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(vector, ep, self.ef_construction, layer)
            m_max = self._slot(layer)

            neighbors = self._select(found, self.M)
            self._set_links(node, layer, neighbors)

            # Links are bidirectional. Adding the reverse edge can push a
            # neighbor past its cap, so re-select that neighbor's best links.
            for n in neighbors:
                n_links, n_counts = self._slots(n)
                n_buf = n_links[layer][: n_counts[layer]]
                if n_buf.size < m_max:
                    n_links[layer][n_buf.size] = node  # the back edge, into a free slot
                    n_counts[layer] = n_buf.size + 1
                    continue
                # Full. The new node is appended first and then competes: the
                # best m_max of the m_max + 1 candidates win. Scoring it in here
                # rather than evicting up front matters -- a node that is already
                # at its cap would otherwise never be able to gain the incoming
                # edge, and the best-connected part of the graph would freeze.
                cand = np.empty(m_max + 1, dtype=np.int32)
                cand[:m_max] = n_buf
                cand[m_max] = node
                self.dist_evals += m_max + 1
                scored = sorted(
                    zip((self.matrix[cand] @ self.matrix[n]).tolist(), cand.tolist()),
                    reverse=True,
                )
                self._set_links(n, layer, self._select(scored, m_max))

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
        query,
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
        q = np.asarray(query, dtype=self.dtype)

        # Greedy descent through the upper layers (beam of 1).
        self.dist_evals += 1
        ep = [(float(q @ self.matrix[self.entry]), self.entry)]
        for layer in range(self.max_level, 0, -1):
            ep = self._search_layer(q, ep, 1, layer)

        # Wide beam search on the full layer 0 graph.
        found = self._search_layer(q, ep, ef, 0)

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
        if target_word not in self.word_to_id:
            raise ValueError(f"{target_word} not in vocabulary")
        return self.search(
            self.matrix[self.word_to_id[target_word]], k, ef, exclude=target_word
        )
