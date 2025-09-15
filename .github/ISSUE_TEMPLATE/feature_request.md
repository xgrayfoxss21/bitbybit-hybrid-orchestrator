---
name: "✨ Feature request"
about: "Propose an enhancement or new capability for BitNet Hybrid Orchestrator"
title: "Feat: <short summary>"
labels: ["enhancement", "design-needed"]
assignees: []
---

<!--
💬 For open-ended ideation or questions, please start in Discussions:
https://github.com/ShiySabiniano/bitnet-hybrid-orchestrator/discussions
Then file this request once there's a concrete proposal.

🔐 Do NOT disclose security-sensitive details here. Use SECURITY.md instead.
-->

## Summary / User story
*A concise description and who benefits.*
> As a <role>, I want <capability> so that <outcome>.

## Motivation / Problem
*Why is this needed? What’s the limitation today? Provide real use cases.*

## Proposed solution
*High-level approach. Include diagrams or links if helpful.*

- Approach A: …
- Approach B: …

## Why now?
*Priority, impact, or deadlines (if any).*

---

## Scope
- [ ] In-scope item 1
- [ ] In-scope item 2

**Out of scope**
- Not included: …

## API / Schema changes (if any)
*Describe changes to `orchestrator/pipeline*.yml` (new nodes, params, policies, conversation fields).*

```yaml
# example
version: 0.1.0
schema: pipeline.v1
nodes:
  - id: <new-node>
    agent: my.agent
    deps: [parse]
    params: { ... }
````

*Runtime interfaces (keep agent contract: `async def agent(**kwargs) -> dict`):*

```python
async def my_agent(**kwargs) -> dict:
    # returns at least {"text": "..."}
    ...
```

## UI / UX (if applicable)

*Changes for the Gradio single-turn UI or Chat mode.*

* Controls / inputs:
* Outputs / formatting:
* Accessibility / i18n:

## Backward compatibility

* [ ] No breaking changes expected
* [ ] Requires migration notes (describe)

## Security & Safety

* Guard interaction (pre/post): thresholds, new labels, PII handling.
* Data paths / logs: ensure **redacted** text only if stored.
* Threat model notes (if executing tools, retrieval, external calls).

## Compliance

If this will be **hosted**:

* [ ] Add/keep **Source** link to the running commit in UI
* [ ] Return `X-AGPL-Source` header or `/source` endpoint
  (See **COMPLIANCE.md**. Do not paste secrets/keys here.)

## Dependencies

*Libraries, model weights, datasets (with licenses).*

* Libs:
* Models:
* Data:

## Risks / Open questions

* Risk 1:
* Open Q:

## Alternatives considered

* Option 1:
* Option 2:

## Metrics / Success criteria

*How we’ll know this worked; include perf/latency/quality targets.*

* KPI 1:
* KPI 2:

## Testing plan

* Unit tests:
* Orchestrator integration (mock agents):
* Colab manual steps (cells to run):
* Edge cases:

## Documentation updates

* [ ] README
* [ ] docs/api.md
* [ ] docs/architecture.md
* [ ] docs/quickstart.md / docs/colab.md / docs/chat.md
* [ ] CHANGELOG.md

## Additional context / References

*Links to prior discussions, PRs, related issues, papers, designs.*

