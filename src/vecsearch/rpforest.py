"""
Random Projection (RP) forest for approximate nearest neighbor search.

Idea in one paragraph
---------------------
Each tree recursively splits the vectors with random hyperplanes until every
leaf holds at most `leaf_size` vectors. Nearby vectors usually land in the same
leaf. One tree misses many true neighbors (they get separated by some split on
the way down), so we build several independent trees. At query time we search
all trees together with a priority queue that explores the branches we were
"closest to falling into" first, collect candidate vectors from the leaves we
reach, and re-score only those candidates with the real metric.

Assumption: every vector is L2-normalized (your loader already does this).
That makes every hyperplane pass through the origin, so a split is just a
normal vector with no offset.
"""

import heapq
import itertools
import math
import random

from vecsearch.metric import Metric


class _Leaf:
    """Bottom of a tree: a plain list of the words stored here."""

    __slots__ = ("words",)

    def __init__(self, words: list[str]) -> None:
        self.words = words


class _Split:
    """Internal node: a hyperplane and two children.

    `normal` is a UNIT vector. For a point x (also unit length):
        sumprod(normal, x) > 0  ->  x is on the `left` side
        otherwise               ->  x is on the `right` side
    Because normal has length 1, that dot product is also the signed distance
    from x to the plane (the "margin"), which the search uses for priorities.
    """

    __slots__ = ("normal", "left", "right")

    def __init__(self, normal: list[float], left, right) -> None:
        self.normal = normal
        self.left = left
        self.right = right


class RPForest:
    MAX_SPLIT_TRIES = 5  # attempts to find a plane that separates the points

    def __init__(
        self,
        data: dict[str, list[float]],
        metric: Metric,
        n_trees: int = 10,
        leaf_size: int = 32,
        seed: int = 0,
    ) -> None:
        self.data = data  # word -> unit vector. Shared, not copied.
        self.metric = metric  # only used to score final candidates
        self.n_trees = n_trees
        self.leaf_size = leaf_size
        self.rng = random.Random(seed)  # seeded so builds are reproducible

        # Counts plane-side dot products done during SEARCH. These are real
        # work that CountingMetric can't see (we call sumprod directly), so the
        # benchmark adds this to the distance count when comparing indexes.
        self.plane_evals = 0

        # Bulk build: every tree gets the same words but draws different random
        # splits from self.rng (it advances between trees), so trees differ.
        words = list(data)
        self.roots = [self._build(words) for _ in range(n_trees)]

    # ------------------------------------------------------------------
    # BUILDING
    # ------------------------------------------------------------------

    def _try_split(self, words: list[str]):
        """Try to split `words` with one random hyperplane.

        Returns (normal, left_words, right_words), or None if no valid split
        was found (e.g. all vectors identical).
        """
        for _ in range(self.MAX_SPLIT_TRIES):
            # 1. Pick two different random points. Sampling data points (not a
            #    random direction) makes the plane adapt to where data lives.
            word_a, word_b = self.rng.sample(words, 2)
            a, b = self.data[word_a], self.data[word_b]

            # 2. The plane bisecting a and b is perpendicular to (a - b).
            #    Since |a| = |b| = 1 it passes through the origin, so no offset.
            normal = [x - y for x, y in zip(a, b)]

            # 3. Scale the normal to length 1 so dot products become margins.
            length = math.sqrt(math.sumprod(normal, normal))
            if length == 0.0:
                continue  # a and b are duplicates: no direction, resample
            normal = [x / length for x in normal]

            # 4. Route every point by the sign of its dot product with normal.
            #    This is equivalent to "is x closer to a or to b?".
            left, right = [], []
            for word in words:
                if math.sumprod(normal, self.data[word]) > 0:
                    left.append(word)
                else:
                    right.append(word)

            # 5. Reject a split that leaves one side empty (recursion would
            #    never terminate). In exact math a and b always end up on
            #    opposite sides, so this only guards float edge cases.
            if left and right:
                return normal, left, right

        return None

    def _build(self, words: list[str]):
        """Recursively build a subtree over `words`. Never mutates `words`."""
        # Small enough: stop splitting. Copy the list, because add() appends to
        # leaf lists and trees must not share them.
        if len(words) <= self.leaf_size:
            return _Leaf(list(words))

        split = self._try_split(words)
        if split is None:
            # Could not split (e.g. many duplicate vectors). Keep an oversized
            # leaf rather than recursing forever.
            return _Leaf(list(words))

        normal, left, right = split
        return _Split(normal, self._build(left), self._build(right))

    # ------------------------------------------------------------------
    # ADDING POINTS
    # ------------------------------------------------------------------

    def add(self, word: str, vector: list[float]) -> None:
        """Insert one new (already L2-normalized) vector into every tree.

        For each tree: walk down using the existing planes, append the word to
        the leaf it lands in, and if that leaf now exceeds leaf_size, split it.
        Existing planes are never changed, so this is cheap, but a tree built
        on old data won't adapt if the distribution shifts a lot. Rebuild
        occasionally if you add many points.
        """
        if word in self.data:
            raise ValueError(f"{word} already in index")

        self.data[word] = vector  # store first so splitting can look it up
        for i, root in enumerate(self.roots):
            # _insert returns the (possibly replaced) subtree root. The root
            # itself changes when a single-leaf tree gets its first split.
            self.roots[i] = self._insert(root, word)

    def _insert(self, node, word: str):
        vector = self.data[word]

        if isinstance(node, _Split):
            # Same routing rule as building and searching.
            if math.sumprod(node.normal, vector) > 0:
                node.left = self._insert(node.left, word)
            else:
                node.right = self._insert(node.right, word)
            return node

        # Reached a leaf: store the word here.
        node.words.append(word)

        # Overfull? Rebuild this leaf as a subtree. _build splits it (possibly
        # several levels) or returns a fresh leaf if it can't be split.
        if len(node.words) > self.leaf_size:
            return self._build(node.words)
        return node

    # ------------------------------------------------------------------
    # SEARCHING
    # ------------------------------------------------------------------

    def search(
        self,
        query: list[float],
        k: int = 10,
        search_k: int | None = None,
        exclude: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return the approximate top-k (word, score) pairs for `query`.

        search_k: how many candidates to collect before stopping. This is the
        recall/speed knob (like `ef` in graph indexes). Bigger = better recall,
        more distance computations. Defaults to k * n_trees like Annoy.
        """
        if search_k is None:
            search_k = k * self.n_trees

        # Best-first search over ALL trees at once. Heap entries are
        # (cost, tiebreak, node). Lowest cost pops first. `cost` is the largest
        # margin we had to "cross" (go against the query's natural side) to
        # reach that node: 0 means we followed the query's side all the way, a
        # small cost means we only crossed planes the query was close to, so
        # neighbors are likely there. The tiebreak counter stops Python from
        # trying to compare node objects when costs are equal.
        tie = itertools.count()
        heap = [(0.0, next(tie), root) for root in self.roots]
        heapq.heapify(heap)

        candidates: set[str] = set()  # set: same word can appear in many trees

        while heap and len(candidates) < search_k:
            cost, _, node = heapq.heappop(heap)

            if isinstance(node, _Leaf):
                # Leaf reached: everything in here is a candidate.
                candidates.update(node.words)
                continue

            # Internal node: which side is the query on, and by how much?
            margin = math.sumprod(node.normal, query)
            self.plane_evals += 1

            if margin > 0:
                near, far = node.left, node.right
            else:
                near, far = node.right, node.left

            # The near side costs nothing extra: it's where the query belongs.
            heapq.heappush(heap, (cost, next(tie), near))
            # The far side is reachable but risky. Its cost is the worst margin
            # crossed so far. A query almost on the plane (tiny |margin|) makes
            # the far side very attractive to explore early.
            heapq.heappush(heap, (max(cost, abs(margin)), next(tie), far))

        # The tree phase only produced SUSPECTS. Now score them exactly with
        # the real metric and keep the best k. This is the only place the
        # injected metric is called, so CountingMetric counts these.
        if exclude is not None:
            candidates.discard(exclude)
        scored = ((w, self.metric(query, self.data[w])) for w in candidates)
        return heapq.nlargest(k, scored, key=lambda pair: pair[1])

    def find_closest(
        self, target_word: str, k: int = 5, search_k: int | None = None
    ) -> list[tuple[str, float]]:
        """Same interface as KNN.find_closest, so the bench harness works."""
        if target_word not in self.data:
            raise ValueError(f"{target_word} not in vocabulary")
        return self.search(self.data[target_word], k, search_k, exclude=target_word)
