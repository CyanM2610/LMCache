# Final two-axis review

Scope: working-tree changes against LMCache v0.5.5
(`05a013b29da78cf2321b9b46ec5039dde2fb0bb0`), including new implementation,
tests and examples. No commits were created. The review used the code-review
skill with independent Standards and Spec agents. Specifications are preserved
under `spec/`, subject to the user's CXLMemSim/existing-version amendment.

## Standards

The documented helper-order and ownership/lifetime docstring findings are
resolved. Public lifecycle methods now state failure propagation and the need
to retain protected resources after failed cleanup. The reviewer confirmed
zero outstanding hard findings after rechecking those edits.

Two optional heuristics remain: numeric native terminal-state values appear
at both the client and completion boundary; allocator teardown mutates public
tensor/buffer fields. Neither was a documented ownership violation or a
concrete correctness finding. The current explicit ownership/release contract
is retained; a broader upstream allocator API change is outside this patch.

Tool-enforced rules are reported separately in the pre-commit logs. Relevant
changed-file checks pass, including mypy; Rust hooks are explicitly skipped.
This does not claim all-files repository CI passed.

## Spec

The review found a P1 cancellation defect: successful lookup could retain
nonexpiring read references if END_SESSION arrived before LOAD. The requirement
to clear references and reservations was violated. The red regression is
`evidence/bootstrap/lookup-cancel-red.log`.

The fix gives each lookup an owned lease. LOAD exchanges its own logical
references for physical pins, and cancellation releases only unused ownership.
No session timeout can silently discard the owner. Late RPCs are rejected
using request-ID tombstones. Submitted work keeps its pins until CUDA/model
completion; another request's reservation cannot be released or borrowed.

The public-path regression, two-reader/pin-pressure test, and real Qwen lookup
cancellation now pass. The real probe reserved and released exactly two
references, submitted no transfer for the cancelled request, left no live
session/job/reader/writer/pin, and retried with exact tokens and logprobs.
The Spec reviewer also checked server-close ordering and late RPCs and found
no unresolved race, duplicate release or premature source reuse in these paths.

Standards: 0 unresolved hard findings, 2 optional heuristics (protocol naming
and teardown coupling). Spec: 0 unresolved findings in reviewed paths; the
original P1 cancellation defect is fixed and reproduced in passing evidence.
