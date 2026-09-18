# Agent Memory Stack: a trustworthy-memory blueprint

This document turns the project from a collection of retrieval experiments
into a system design. It is deliberately a blueprint, not a claim that every
layer below is already implemented.

## Goal and boundary

The target is a memory subsystem that retains useful, revisable evidence for
an agent while making ownership, provenance, safety state, and historical
change inspectable. It complements a RAG system and a model's context window;
it does not claim to replace either, perform autonomous weight training, or
solve AGI.

## Logical architecture

```text
                               trusted host / MCP adapter (future)
                                             |
    applications --> [ memory decision / context routing ] <--- outcome feedback
                                             |
               [ proactive consolidation and conflict review ]       (future)
                                             |
                       [ sharing policy / privacy boundary ]
                                             |
                      [ current projection + event archive ]
                                             |
            [ guarded write + provenance + trust + visibility ]  <-- since v0.4
                                             |
           [ text / code / image artifact adapters and extraction ]   (future)
```

Feedback is a cross-layer loop, not a storage layer: task outcomes can inform
future retention, review, and routing policies, but must never silently alter
the original evidence. MCP is likewise an adapter at the boundary; it should
transport the record semantics below, not define them.

## Core record contract

Every accepted memory event or current projection uses the following logical
fields. A single `IgmStore` implements both halves today: the event/current
portions through archive supersede (the default), and provenance, visibility,
and trust for new records.

| Field | Meaning | 0.5 |
|---|---|---|
| `memoryId` | Stable record identifier | implemented |
| `text` | Retained source statement | implemented (text only) |
| `slot` | Attribute used for supersede/current projection | implemented |
| `scope` | `user` or `project` storage ownership | implemented |
| `visibility` | `private`, `project`, or explicit `cross-project` | implemented |
| `provenance` | Source, actor, observed time, optional reference | implemented |
| `trust` | `accepted`, `review`, or `rejected` plus reasons/timestamps | implemented |
| `eventId`, `validFrom`, `validTo`, `supersedes` | Version timeline | implemented (default) |
| `artifact` | Image/code/file reference, checksum, modality, extractor | future contract |
| `outcomeEvidence` | Later use/verification signals | future contract |

The associated `CurrentProjection` contains only accepted events. A candidate
with `trust: review` is stored outside that projection, so it cannot replace a
known-good current value just by being written.

## Implemented vertical slice

1. A caller proposes text with optional `cwd`, `provenance`, and `visibility`.
2. The importance gate rejects non-durable content as before.
3. The deterministic guard rejects credential-shaped values without retaining
   their text, or quarantines high-signal control-instruction patterns.
4. An accepted item is appended to the event archive and closes the `validTo`
   of the value it supersedes; `supersede: "delete"` opts out into the
   destructive current-only store.
5. Only accepted items reach query, history, prompt injection, and matching
   cross-project retrieval. Cross-project retrieval additionally requires a
   project `experience` explicitly marked `cross-project`.
6. A trusted host audits a redacted record and may accept or reject a pending
   candidate. Model-facing tools cannot perform that approval.

## Trustworthy-memory dimensions

| Dimension | Stack responsibility | Status |
|---|---|---|
| Integrity | Modality/artifact capture preserves source identity | text provenance now; artifacts future |
| Security | Guarded write, quarantine, safe read/injection | MVP implemented |
| Privacy | Scope and explicit sharing policy | single-profile/project MVP implemented |
| Explainability | Provenance, trust reason, review timestamps | MVP implemented |
| Auditability | Versioned events and redacted write audit | MVP implemented |
| Adaptivity | Outcome-driven retention/review policy | future |
| Interoperability | MCP transport of the record contract | future |
| Efficiency | Memory-vs-context routing and storage indexes | future |

## Near-term validation plan

The next valid experiment is not a generic “beats RAG” benchmark. Construct a
small adversarial memory suite with known expected outcomes:

| Scenario | Expected invariant |
|---|---|
| Normal current-state update | New accepted value supersedes old value; old event remains in the archive |
| Prompt-control write | Candidate is not queryable, injected, shared, or allowed to replace old state before review |
| Credential-shaped write | Secret text is absent from persisted JSON; fingerprint audit exists |
| Cross-project lesson | Project-only experience stays isolated; explicitly shared, topic-matched experience can transfer |
| Host approval | Only an explicit host review promotes a quarantined candidate into normal version/slot behavior |

These scenarios are covered by the repository's regression tests. They verify
the stated rule boundary, not recall quality against a public adversarial
benchmark or real-world compromise resistance.

## Sequencing

1. Harden this trust layer with corpus-based false-positive/false-negative
   measurement and a real review UI.
2. Add artifact references for code/files first; add image embeddings only
   alongside a chosen model and an evaluation dataset.
3. Define an MCP server that exposes the same scope/visibility/trust contract.
4. Add multi-agent identity, authorization, and conflict resolution before
   enabling broad sharing.
5. Measure context-memory routing and feedback policies on real user tasks.
