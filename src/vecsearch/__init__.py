from vecsearch.io import load_normalized_data
from vecsearch.knn import KNN
from vecsearch.metric import CosineSimilarityMetric


DATAPATH = "data/glove.6B.100d.txt"
PERCENTAGE = 1.0


def main() -> None:
    data = load_normalized_data(DATAPATH, PERCENTAGE)
    print("Data loaded")

    knn = KNN(data, CosineSimilarityMetric())
    print("KNN Constructed")

    k_value = int(input("How many?: "))
    while True:
        target_word = input("Word: ").lower()

        try:
            result = knn.find_closest(target_word, k_value)
        except ValueError as err:
            print(err)
            continue

        for word in result:
            print(f"{word}")
