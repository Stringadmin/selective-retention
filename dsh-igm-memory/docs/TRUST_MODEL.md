# Trust model and guarded-write policy

## Assets and adversaries

The protected asset is the model-visible durable-memory path: values returned
by query/history, added to the system prompt, or offered as a cross-project
lesson. The v0.4 guard assumes an untrusted user message, imported string, or
agent-proposed memory can try to place a control instruction on that path.

It does not assume the local host filesystem, all plugin services, or an
already-compromised DSH process are adversarial. Anyone able to read the JSON
file can inspect its accepted values and quarantined non-secret candidate text.

## Policy

| Input classification | Stored text | Model-visible | Audit |
|---|---:|---:|---:|
| Normal durable statement | yes | yes | normal metadata |
| Credential-shaped statement | no | no | fingerprint only |
| High-signal control instruction | quarantine only | no | redacted candidate metadata |
| Host-approved candidate | yes, via ordinary storage | yes | review decision retained |
| Host-rejected candidate | quarantine only | no | review decision retained |

The classifier is a deterministic regular-expression policy. It is designed
to create a dependable, testable floor, not to infer intent. A false positive
is recoverable through trusted host review; a false negative remains possible.

## Invariants

1. A `review` candidate does not alter `items`, `events`, or the current slot
   projection before approval.
2. Retrieval, prompt injection, history, and cross-project sharing filter for
   `trust: accepted`.
3. Cross-project transfer additionally requires `scope: project`,
   `type: experience`, `visibility: cross-project`, and topic overlap.
4. `igm.memory.review` is a host service, not a model-facing tool.
5. Credential rejection persists a SHA-256-derived short fingerprint and
   metadata only; the submitted credential is never written to the store.

Pre-0.4 project experiences had implicit cross-project behavior. They load as
legacy shared experiences to avoid silently changing existing deployments.
For a strict boundary, audit and rewrite those records with explicit
`visibility: project` before enabling sharing in production.

## Non-goals

- This is not a sandbox, malware scanner, authentication system, encryption
  layer, content classifier, or complete prompt-injection defense.
- It does not protect memories written by another plugin or direct file edits.
- `securityEnabled: false` disables this guard and is appropriate only for
  controlled compatibility testing.
- Approval transfers responsibility to the trusted host. Do not approve a
  candidate simply because it is syntactically a memory statement.

## Reviewer guidance

Use the redacted audit list to locate a `reviewId`. Inspect raw text only in a
trusted host UI/process, decide whether it is a benign quote or a legitimate
fact, and then call either `accept` or `reject`. When in doubt, reject and
recreate the useful fact in a concise, non-instructional form.
