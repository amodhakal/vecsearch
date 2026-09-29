from collections import defaultdict
import math


def load_normalized_data(filepath: str, percent=1.0):
    with open(filepath, "r", encoding="utf-8") as f:
        result: dict[str, list[float]] = defaultdict(list[float])

        total_lines = sum(1 for line in f if line.strip())
        max_lines = int(total_lines * percent)
        f.seek(0)

        for line in f:
            if len(result) >= max_lines:
                break

            parts = line.strip().split()
            if not parts:
                continue

            word = parts[0]
            vector = [float(x) for x in parts[1:]]
            normal = math.sqrt(sum(x * x for x in vector))
            if normal == 0:
                continue

            vector = [x / normal for x in vector]
            result[word] = vector

    return result
