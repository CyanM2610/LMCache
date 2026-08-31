# SPDX-License-Identifier: Apache-2.0

"""Global HotPrefix radix and fleet-wide hotness state."""

# Standard
from dataclasses import dataclass, field
import hashlib

PrefixId = bytes


@dataclass(frozen=True)
class PrefixAccessObservation:
    """One instance's initial request lookup observation.

    Args:
        instance_id: Non-negative vLLM instance identity.
        local_event_seq: Positive sequence number unique within the instance.
        token_ids: Complete request token path.
        matched_tokens: Tokens reported by the instance's native local APC.
    """

    instance_id: int
    local_event_seq: int
    token_ids: tuple[int, ...]
    matched_tokens: int

    def __post_init__(self) -> None:
        if self.instance_id < 0:
            raise ValueError("instance_id must be non-negative")
        if self.local_event_seq <= 0:
            raise ValueError("local_event_seq must be positive")
        if not self.token_ids or any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("token_ids must contain non-negative IDs")
        if self.matched_tokens < 0 or self.matched_tokens > len(self.token_ids):
            raise ValueError("matched_tokens is outside the token range")


@dataclass(frozen=True)
class PrefixAccessResult:
    """Idempotent result of committing one prefix access observation."""

    epoch: int
    path: tuple[PrefixId, ...]
    global_matched_tokens: int


@dataclass(frozen=True)
class GlobalPrefixNodeSnapshot:
    """Immutable view of one Global Host Prefix Tree node."""

    prefix_id: PrefixId
    full_prefix: tuple[int, ...]
    segment: tuple[int, ...]
    parent: PrefixId | None
    children: tuple[PrefixId, ...]
    frequency: int
    clock: int
    depth: int

    @property
    def global_hotness(self) -> int:
        """Return the fleet-wide Host retention score."""
        return self.frequency * self.clock


@dataclass
class _Node:
    segment: tuple[int, ...]
    full_prefix: tuple[int, ...]
    prefix_id: PrefixId
    depth: int
    frequency: int
    clock: int
    parent: "_Node | None"
    children: dict[int, "_Node"] = field(default_factory=dict)


class GlobalHostPrefixTree:
    """Merge instance access streams into one shared-tier HotPrefix view.

    Args:
        namespace: Stable model/cache namespace included in prefix identities.
        max_value: Saturating upper bound for frequency and depth.
        max_age: Clock assigned to accessed and newly observed nodes.
        aging_interval: Committed observations between global aging passes.
    """

    def __init__(
        self,
        *,
        namespace: bytes,
        max_value: int = 255,
        max_age: int = 255,
        aging_interval: int = 50,
    ) -> None:
        if max_value <= 0:
            raise ValueError("max_value must be positive")
        if max_age < 0 or max_age > max_value:
            raise ValueError("max_age must be between zero and max_value")
        if aging_interval <= 0:
            raise ValueError("aging_interval must be positive")
        self._namespace = namespace
        self._max_value = max_value
        self._max_age = max_age
        self._aging_interval = aging_interval
        self._epoch = 0
        self._requests_since_aging = 0
        self._root = _Node((), (), b"", 0, 0, 0, None)
        self._nodes_by_id: dict[PrefixId, _Node] = {}
        self._results: dict[tuple[int, int], PrefixAccessResult] = {}
        self._last_local_seq: dict[int, int] = {}

    def observe(self, observation: PrefixAccessObservation) -> PrefixAccessResult:
        """Commit one access exactly once and return its global epoch.

        Args:
            observation: Initial request access emitted by one instance.

        Returns:
            The original result for duplicate events or a newly committed result.

        Raises:
            ValueError: If an unseen event arrives behind an instance's sequence.
        """
        event_id = (observation.instance_id, observation.local_event_seq)
        existing = self._results.get(event_id)
        if existing is not None:
            return existing
        last_seq = self._last_local_seq.get(observation.instance_id, 0)
        if observation.local_event_seq <= last_seq:
            raise ValueError("unseen PrefixAccessObservation arrived out of order")

        path, matched_tokens = self._observe_path(observation.token_ids)
        self._epoch += 1
        self._requests_since_aging += 1
        if self._requests_since_aging == self._aging_interval:
            self._age_all()
            self._requests_since_aging = 0
        result = PrefixAccessResult(
            self._epoch,
            tuple(node.prefix_id for node in path),
            matched_tokens,
        )
        self._results[event_id] = result
        self._last_local_seq[observation.instance_id] = observation.local_event_seq
        return result

    def snapshot(self) -> tuple[GlobalPrefixNodeSnapshot, ...]:
        """Return all global logical nodes in deterministic prefix order."""
        nodes: list[_Node] = []
        stack = list(self._root.children.values())
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
        nodes.sort(key=lambda item: item.full_prefix)
        return tuple(self._snapshot_node(node) for node in nodes)

    def get(self, prefix_id: PrefixId) -> GlobalPrefixNodeSnapshot | None:
        """Return the current authoritative Global Hotness for one prefix.

        Args:
            prefix_id: Canonical LogicalPrefix identifier.

        Returns:
            The current node snapshot, or ``None`` when it is unknown.
        """
        node = self._nodes_by_id.get(prefix_id)
        return None if node is None else self._snapshot_node(node)

    def _observe_path(self, token_ids: tuple[int, ...]) -> tuple[list[_Node], int]:
        remaining = token_ids
        node = self._root
        path: list[_Node] = []
        matched_tokens = 0
        while remaining:
            child = node.children.get(remaining[0])
            if child is None:
                child = self._new_node(node, remaining)
                node.children[remaining[0]] = child
                path.append(child)
                break
            common = self._common_prefix_length(child.segment, remaining)
            if common < len(child.segment):
                split_parent = self._split_node(child, common)
                self._access(split_parent)
                path.append(split_parent)
                matched_tokens += common
                remaining = remaining[common:]
                if remaining:
                    new_node = self._new_node(split_parent, remaining)
                    split_parent.children[remaining[0]] = new_node
                    path.append(new_node)
                break
            self._access(child)
            path.append(child)
            matched_tokens += common
            node = child
            remaining = remaining[common:]
        return path, matched_tokens

    def _new_node(self, parent: _Node, segment: tuple[int, ...]) -> _Node:
        full_prefix = parent.full_prefix + segment
        node = _Node(
            segment,
            full_prefix,
            self._make_prefix_id(full_prefix),
            min(parent.depth + 1, self._max_value),
            1,
            self._max_age,
            parent,
        )
        self._nodes_by_id[node.prefix_id] = node
        return node

    def _split_node(self, child: _Node, split_length: int) -> _Node:
        if split_length <= 0 or split_length >= len(child.segment):
            raise ValueError("split_length must be inside a node segment")
        parent = child.parent
        if parent is None:
            raise RuntimeError("root node cannot be split")
        parent_segment = child.segment[:split_length]
        parent_prefix = parent.full_prefix + parent_segment
        split_parent = _Node(
            parent_segment,
            parent_prefix,
            self._make_prefix_id(parent_prefix),
            child.depth,
            child.frequency,
            child.clock,
            parent,
        )
        self._nodes_by_id[split_parent.prefix_id] = split_parent
        parent.children[parent_segment[0]] = split_parent
        child.segment = child.segment[split_length:]
        child.parent = split_parent
        split_parent.children[child.segment[0]] = child
        self._update_depths(child, split_parent.depth + 1)
        return split_parent

    def _update_depths(self, node: _Node, depth: int) -> None:
        node.depth = min(depth, self._max_value)
        for child in node.children.values():
            self._update_depths(child, depth + 1)

    def _access(self, node: _Node) -> None:
        node.frequency = min(node.frequency + 1, self._max_value)
        node.clock = self._max_age

    def _age_all(self) -> None:
        stack = list(self._root.children.values())
        while stack:
            node = stack.pop()
            node.clock = max(0, node.clock - 1)
            stack.extend(node.children.values())

    def _snapshot_node(self, node: _Node) -> GlobalPrefixNodeSnapshot:
        children = tuple(
            child.prefix_id
            for child in sorted(
                node.children.values(), key=lambda item: item.full_prefix
            )
        )
        parent = node.parent
        parent_prefix_id = (
            parent.prefix_id
            if parent is not None and parent is not self._root
            else None
        )
        return GlobalPrefixNodeSnapshot(
            node.prefix_id,
            node.full_prefix,
            node.segment,
            parent_prefix_id,
            children,
            node.frequency,
            node.clock,
            node.depth,
        )

    def _make_prefix_id(self, full_prefix: tuple[int, ...]) -> PrefixId:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(len(self._namespace).to_bytes(4, "little"))
        digest.update(self._namespace)
        for token_id in full_prefix:
            digest.update(token_id.to_bytes(8, "little", signed=False))
        return digest.digest()

    def _common_prefix_length(
        self, left: tuple[int, ...], right: tuple[int, ...]
    ) -> int:
        common = 0
        for left_token, right_token in zip(left, right, strict=False):
            if left_token != right_token:
                break
            common += 1
        return common
