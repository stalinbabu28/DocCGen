from __future__ import annotations

from evaluation.test_cases_50 import TESTS
from pipeline.select_document import select_document


def _selected_module_from_path(path: str) -> str:
    return (
        path.split("/")[-1]
        .replace("azure.azcollection.", "")
        .replace(".json", "")
    )


def main() -> None:
    correct = 0

    for sample in TESTS:
        query = sample["query"]
        expected = sample["expected_module"]

        selected = select_document(query)
        module = _selected_module_from_path(selected)

        print("\n" + "=" * 80)
        print("Query:", query)
        print("Expected:", expected)
        print("Selected:", module)

        if module == expected:
            print("CORRECT")
            correct += 1
        else:
            print("WRONG")

    accuracy = correct / len(TESTS)

    print("\n" + "=" * 80)
    print("Selector Accuracy:", round(accuracy * 100, 2), "%")


if __name__ == "__main__":
    main()