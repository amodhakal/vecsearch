from vecsearch.metric import Metric
import heapq


class KNN:
    def __init__(self, data: dict[str, list[float]], metric: Metric) -> None:
        self.data = data
        self.metric = metric

    def find_closest(self, target_word: str, k=5):
        if target_word not in self.data:
            raise ValueError(f"{target_word} not in vocabulary")

        target_vec = self.data[target_word]
        scores = (
            (word, self.metric(target_vec, vec))
            for word, vec in self.data.items()
            if word != target_word
        )

        return heapq.nlargest(k, scores, key=lambda x: x[1])
