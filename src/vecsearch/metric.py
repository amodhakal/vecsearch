from abc import ABC, abstractmethod
import math


class Metric(ABC):
    @abstractmethod
    def __call__(self, vec1: list[float], vec2: list[float]) -> float: ...


class CosineSimilarityMetric(Metric):
    def __call__(self, vec1, vec2):
        return math.sumprod(vec1, vec2)
