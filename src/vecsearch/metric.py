from abc import ABC, abstractmethod


class Metric(ABC):
    @abstractmethod
    def __call__(self, vec1: list[float], vec2: list[float]) -> float: ...


class CosineSimilarityMetric(Metric):
    def __call__(self, vec1, vec2):
        return sum(a * b for a, b in zip(vec1, vec2))
