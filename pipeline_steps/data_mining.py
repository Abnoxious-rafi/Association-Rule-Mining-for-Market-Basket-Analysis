"""Stage 6: implement and compare Apriori and FP-Growth."""

from __future__ import annotations

import gc
import itertools
import math
import random
import time
import tracemalloc
from collections import Counter
from collections.abc import Callable, Iterable, Sequence

import pandas as pd


Transaction = frozenset[str]
Itemsets = dict[frozenset[str], float]


def apriori(
    transactions: Sequence[Transaction],
    min_support: float,
    max_length: int | None,
) -> Itemsets:
    """Mine frequent itemsets with the level-wise Apriori algorithm."""
    transaction_count = len(transactions)
    minimum_count = math.ceil(min_support * transaction_count - 1e-12)
    item_counts: Counter[str] = Counter()
    for transaction in transactions:
        item_counts.update(transaction)

    current: dict[frozenset[str], int] = {
        frozenset((item,)): count
        for item, count in item_counts.items()
        if count >= minimum_count
    }
    found = dict(current)
    frequent_single_items = {next(iter(itemset)) for itemset in current}
    itemset_length = 1

    while current and (max_length is None or itemset_length < max_length):
        previous_tuples = sorted(tuple(sorted(itemset)) for itemset in current)
        previous_set = set(current)
        candidates: set[tuple[str, ...]] = set()

        for left_index, left in enumerate(previous_tuples):
            for right in previous_tuples[left_index + 1 :]:
                if left[:-1] != right[:-1]:
                    if right[:-1] > left[:-1]:
                        break
                    continue
                candidate_tuple = tuple(sorted(set(left).union(right)))
                if len(candidate_tuple) != itemset_length + 1:
                    continue
                if all(
                    frozenset(subset) in previous_set
                    for subset in itertools.combinations(
                        candidate_tuple, itemset_length
                    )
                ):
                    candidates.add(candidate_tuple)

        if not candidates:
            break

        candidate_counts: Counter[tuple[str, ...]] = Counter()
        for transaction in transactions:
            eligible = sorted(transaction.intersection(frequent_single_items))
            for combination in itertools.combinations(
                eligible, itemset_length + 1
            ):
                if combination in candidates:
                    candidate_counts[combination] += 1

        current = {
            frozenset(items): count
            for items, count in candidate_counts.items()
            if count >= minimum_count
        }
        found.update(current)
        itemset_length += 1

    return {
        itemset: count / transaction_count for itemset, count in found.items()
    }


class _FPNode:
    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(
        self, item: str | None, parent: _FPNode | None = None
    ) -> None:
        self.item = item
        self.count = 0
        self.parent = parent
        self.children: dict[str, _FPNode] = {}
        self.link: _FPNode | None = None


class _HeaderEntry:
    __slots__ = ("count", "head", "tail")

    def __init__(self, count: int) -> None:
        self.count = count
        self.head: _FPNode | None = None
        self.tail: _FPNode | None = None


def _build_fp_tree(
    weighted_transactions: Iterable[tuple[Iterable[str], int]],
    minimum_count: int,
) -> tuple[_FPNode | None, dict[str, _HeaderEntry]]:
    """Build an FP-tree from transactions that may carry multiplicities."""
    transactions = [(tuple(items), weight) for items, weight in weighted_transactions]
    item_counts: Counter[str] = Counter()
    for items, weight in transactions:
        for item in set(items):
            item_counts[item] += weight

    frequent_counts = {
        item: count
        for item, count in item_counts.items()
        if count >= minimum_count
    }
    if not frequent_counts:
        return None, {}

    order = {
        item: rank
        for rank, (item, _) in enumerate(
            sorted(frequent_counts.items(), key=lambda pair: (-pair[1], pair[0]))
        )
    }
    header = {
        item: _HeaderEntry(count) for item, count in frequent_counts.items()
    }
    root = _FPNode(None)

    for items, weight in transactions:
        ordered_items = sorted(
            (item for item in set(items) if item in frequent_counts),
            key=order.__getitem__,
        )
        current = root
        for item in ordered_items:
            child = current.children.get(item)
            if child is None:
                child = _FPNode(item, current)
                current.children[item] = child
                entry = header[item]
                if entry.head is None:
                    entry.head = child
                else:
                    assert entry.tail is not None
                    entry.tail.link = child
                entry.tail = child
            child.count += weight
            current = child

    return root, header


def fp_growth(
    transactions: Sequence[Transaction],
    min_support: float,
    max_length: int | None,
) -> Itemsets:
    """Mine frequent itemsets with FP-Growth and conditional FP-trees."""
    transaction_count = len(transactions)
    minimum_count = math.ceil(min_support * transaction_count - 1e-12)
    root, header = _build_fp_tree(
        ((transaction, 1) for transaction in transactions), minimum_count
    )
    found_counts: dict[frozenset[str], int] = {}
    if root is None:
        return {}

    def mine_tree(
        current_header: dict[str, _HeaderEntry],
        prefix: frozenset[str],
    ) -> None:
        mining_order = sorted(
            current_header.items(), key=lambda pair: (pair[1].count, pair[0])
        )
        for item, entry in mining_order:
            pattern = prefix.union((item,))
            found_counts[pattern] = entry.count
            if max_length is not None and len(pattern) >= max_length:
                continue

            conditional_patterns: list[tuple[tuple[str, ...], int]] = []
            node = entry.head
            while node is not None:
                path: list[str] = []
                parent = node.parent
                while parent is not None and parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent
                if path:
                    conditional_patterns.append((tuple(path), node.count))
                node = node.link

            _, conditional_header = _build_fp_tree(
                conditional_patterns, minimum_count
            )
            if conditional_header:
                mine_tree(conditional_header, pattern)

    mine_tree(header, frozenset())
    return {
        itemset: count / transaction_count
        for itemset, count in found_counts.items()
    }


def generate_rules(
    itemsets: Itemsets,
    min_confidence: float,
) -> list[dict[str, object]]:
    """Generate rules with the requested confidence and positive-lift filters."""
    rules: list[dict[str, object]] = []
    for itemset, support in itemsets.items():
        if len(itemset) < 2:
            continue
        ordered_items = sorted(itemset)
        for antecedent_size in range(1, len(ordered_items)):
            for antecedent_values in itertools.combinations(
                ordered_items, antecedent_size
            ):
                antecedent = frozenset(antecedent_values)
                consequent = itemset.difference(antecedent)
                antecedent_support = itemsets.get(antecedent)
                consequent_support = itemsets.get(consequent)
                if antecedent_support is None or consequent_support is None:
                    raise RuntimeError(
                        "A frequent itemset is missing a required subset."
                    )

                confidence = support / antecedent_support
                lift = confidence / consequent_support
                if confidence >= min_confidence and lift > 1:
                    rules.append(
                        {
                            "antecedents": antecedent,
                            "consequents": consequent,
                            "support": support,
                            "confidence": confidence,
                            "lift": lift,
                        }
                    )
    return sorted(
        rules,
        key=lambda rule: (
            -float(rule["lift"]),
            -float(rule["confidence"]),
            -float(rule["support"]),
        ),
    )


def _profile_miner(
    miner: Callable[
        [Sequence[Transaction], float, int | None], Itemsets
    ],
    transactions: Sequence[Transaction],
    min_support: float,
    max_length: int | None,
) -> tuple[Itemsets, float, float]:
    """Measure miner runtime and Python peak allocations, excluding input data."""
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    try:
        itemsets = miner(transactions, min_support, max_length)
        elapsed = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return itemsets, elapsed, peak_bytes / (1024 * 1024)


def _assert_same_itemsets(
    left: Itemsets, right: Itemsets, context: str
) -> None:
    if left.keys() != right.keys():
        only_left = len(left.keys() - right.keys())
        only_right = len(right.keys() - left.keys())
        raise RuntimeError(
            f"Apriori and FP-Growth disagree for {context}: "
            f"{only_left} Apriori-only and {only_right} FP-Growth-only itemsets."
        )
    for itemset in left:
        if not math.isclose(left[itemset], right[itemset], rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(
                f"Apriori and FP-Growth have different support for "
                f"{sorted(itemset)} in {context}."
            )


def mine_data(
    baskets: list[Transaction],
    supports: Sequence[float],
    fixed_support: float,
    dataset_sizes: Sequence[float],
    repeats: int,
    seed: int,
    max_length: int | None,
    min_confidence: float,
) -> tuple[pd.DataFrame, Itemsets, dict[str, object]]:
    """Run support-sensitivity and dataset-size experiments."""
    rng = random.Random(seed)
    randomized_indices = list(range(len(baskets)))
    rng.shuffle(randomized_indices)

    cases: dict[tuple[float, tuple[int, ...]], dict[str, object]] = {}

    def add_case(
        setting: str, support: float, percentage: float, indices: tuple[int, ...]
    ) -> None:
        key = (support, indices)
        if key not in cases:
            cases[key] = {"descriptors": []}
        descriptors = cases[key]["descriptors"]
        assert isinstance(descriptors, list)
        descriptors.append(
            {
                "setting": setting,
                "support": support,
                "dataset_pct": percentage,
                "transaction_count": len(indices),
            }
        )

    all_indices = tuple(range(len(baskets)))
    for support in supports:
        add_case("support_sensitivity", support, 100.0, all_indices)
    for percentage in dataset_sizes:
        sample_count = max(1, math.ceil(len(baskets) * percentage))
        sample_indices = tuple(sorted(randomized_indices[:sample_count]))
        add_case("dataset_size", fixed_support, percentage * 100, sample_indices)

    algorithms = {"Apriori": apriori, "FP-Growth": fp_growth}
    performance_rows: list[dict[str, object]] = []
    baseline_itemsets: Itemsets | None = None
    baseline_rules: list[dict[str, object]] | None = None
    baseline_key = (fixed_support, all_indices)

    for case_number, ((support, indices), case) in enumerate(cases.items(), 1):
        transactions = [baskets[index] for index in indices]
        context = (
            f"case {case_number}/{len(cases)} "
            f"(support={support:g}, baskets={len(transactions):,})"
        )
        print(f"  Mining {context}")
        timings: dict[str, list[float]] = {name: [] for name in algorithms}
        peaks: dict[str, list[float]] = {name: [] for name in algorithms}
        reference_itemsets: Itemsets | None = None

        for repetition in range(repeats):
            names = list(algorithms)
            if repetition % 2:
                names.reverse()
            for name in names:
                itemsets, elapsed, peak_mb = _profile_miner(
                    algorithms[name],
                    transactions,
                    support,
                    max_length,
                )
                timings[name].append(elapsed)
                peaks[name].append(peak_mb)
                if reference_itemsets is None:
                    reference_itemsets = itemsets
                else:
                    _assert_same_itemsets(
                        reference_itemsets,
                        itemsets,
                        f"{context}, repeat {repetition + 1}",
                    )

        if reference_itemsets is None:
            raise RuntimeError(f"No mining result was produced for {context}.")
        generated_rules = generate_rules(reference_itemsets, min_confidence)

        if (support, indices) == baseline_key:
            baseline_itemsets = reference_itemsets
            baseline_rules = generated_rules

        descriptors = case["descriptors"]
        assert isinstance(descriptors, list)
        for descriptor in descriptors:
            for algorithm_name in algorithms:
                elapsed_values = timings[algorithm_name]
                performance_rows.append(
                    {
                        **descriptor,
                        "algorithm": algorithm_name,
                        "mean_run_seconds": sum(elapsed_values) / repeats,
                        "std_run_seconds": (
                            pd.Series(elapsed_values).std(ddof=1)
                            if repeats > 1
                            else 0.0
                        ),
                        "mean_peak_memory_mb": sum(peaks[algorithm_name]) / repeats,
                        "frequent_itemsets": len(reference_itemsets),
                        "association_rules": len(generated_rules),
                        "repeats": repeats,
                        "seed": seed,
                        "max_itemset_length": max_length,
                    }
                )

    if baseline_itemsets is None or baseline_rules is None:
        raise RuntimeError(
            "The fixed-support full-dataset case was not produced. "
            "Include the full dataset size in --sizes."
        )

    performance = pd.DataFrame(performance_rows).sort_values(
        ["setting", "dataset_pct", "support", "algorithm"]
    )
    print(
        "[6/9] Data Mining: compared Apriori and FP-Growth across "
        f"{len(cases)} unique experiment cases"
    )
    return performance, baseline_itemsets, {
        "baseline_rules": baseline_rules,
        "baseline_support": fixed_support,
        "baseline_transaction_count": len(baskets),
    }


def crosscheck_with_mlxtend(
    baskets: Sequence[Transaction],
    min_support: float,
    max_length: int | None,
    expected: Itemsets,
) -> dict[str, int]:
    """Verify one full-data result with MLxtend's independent implementations."""
    from mlxtend.frequent_patterns import apriori as mlxtend_apriori
    from mlxtend.frequent_patterns import fpgrowth as mlxtend_fp_growth
    from mlxtend.preprocessing import TransactionEncoder

    encoder = TransactionEncoder()
    encoded_transactions = [sorted(transaction) for transaction in baskets]
    sparse_matrix = encoder.fit(encoded_transactions).transform(
        encoded_transactions, sparse=True
    )
    encoded = pd.DataFrame.sparse.from_spmatrix(
        sparse_matrix.astype("int8"), columns=encoder.columns_
    ).astype(pd.SparseDtype(bool, False))

    checked: dict[str, int] = {}
    for name, miner in (
        ("MLxtend Apriori", mlxtend_apriori),
        ("MLxtend FP-Growth", mlxtend_fp_growth),
    ):
        result = miner(
            encoded,
            min_support=min_support,
            use_colnames=True,
            max_len=max_length,
        )
        actual = {
            frozenset(items): float(support)
            for items, support in zip(result["itemsets"], result["support"])
        }
        _assert_same_itemsets(expected, actual, name)
        checked[name] = len(actual)
    return checked
