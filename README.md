# PlagueShield

Multi-agent decision support for suspected *Yersinia pestis* cases.

Ten specialist agents read the same case record and continuously answer three
questions:

1. **Is this actually plague?**
2. **Is the isolate showing evidence of unusual antimicrobial resistance?**
3. **What additional test would most reduce uncertainty right now?**

Results are posted to a public read-only status site. The agents run on the
backend; the website shows current case status, each agent's latest work, and
the live research loop without exposing operator controls.

> **Decision support only.** PlagueShield does not prescribe, withhold or modify
> treatment. It is an experimental agentic research system, not a medical device, and is not
> validated for clinical use. Every output requires review by a qualified
> clinician or microbiologist.

---

## Research Results

`/results` is the permanent research-paper archive. Each completed backend
iteration automatically creates a separate article covering its question,
provenance, recorded setup, methods, all agent findings, evidence ablations,
posterior-odds sensitivity, information-gain rankings, GPT-5.5 interpretation,
limitations, and linked references. Articles are automated experimental research
reports, not peer-reviewed papers or clinically validated findings.

The archive is stored in `server/results.sqlite3`. Article identifiers derive
from the source record's SHA-256 hash; reprocessing a record never replaces its
published article. Eligible prior backend runs are backfilled at startup. The
downloadable JSON appendix preserves the exact recorded inputs, prompts,
structured outputs, and timings, including missing/abstained contributions.
Article generation reuses the recorded LLM analysis and makes no additional API
calls. Publicly submitted assessments are not automatically published as papers.

Read-only endpoints: `/api/results`, `/api/results/{id}`, and
`/api/results/{id}/artifact`. The index supports `q`, `origin`, `limit`, and
`offset` parameters. The website includes search, provenance filters, permanent
article URLs, section navigation, and artifact downloads.

## X Publishing

A separate `x_publisher` agent checks the newest completed backend research run
every 300 seconds. It composes one five-post thread covering all ten agents,
labels the case provenance, and links back to the public website. Unchanged
thread content is skipped. It uses the X API, not browser automation, and does
not like, follow, DM, or reply to other accounts.

Configure the new section at the top of `.env`:

```dotenv
X_API_KEY=your_app_api_key
X_API_KEY_SECRET=your_app_api_secret
X_ACCESS_TOKEN=your_account_access_token
X_ACCESS_TOKEN_SECRET=your_account_access_token_secret
PLAGUESHIELD_X_ACCOUNT=plagueshieldX
PLAGUESHIELD_PUBLIC_URL=https://your-public-site.example
PLAGUESHIELD_X_ENABLED=true
PLAGUESHIELD_X_DRY_RUN=true
PLAGUESHIELD_X_INTERVAL_SECONDS=300
```

Use an X developer app with **Read and Write** user permissions and OAuth 1.0a
credentials authorized for the account above. App-only bearer tokens cannot
publish as a user. The publisher verifies `/2/users/me` before posting.
See the [official X posting quickstart](https://docs.x.com/x-api/posts/manage-tweets/quickstart).
Review the draft at `/agents/x_publisher`, then set
`PLAGUESHIELD_X_DRY_RUN=false` and restart the backend to enable real posts.
There is a full five-minute delay before the first scheduled publication.
Changes to `.env` require a backend restart.

Publishing defaults to disabled and dry-run. Credentials are never sent to the
public website. An enabled publisher may create up to 1,440 posts per day if
every cycle contains fresh findings; check your API access, billing budget,
and [X automation rules](https://help.x.com/en/rules-and-policies/x-automation)
before enabling it. Label the X account as automated and disclose its operator.

The durable outbox is `server/x_outbox.sqlite3`. Confirmed post IDs survive
restarts; rate limits delay retries, and partial threads resume from the last
confirmed post. Unconfirmed sends or rejected account credentials pause live
publishing rather than risk duplicates. For recovery, stop the backend, inspect
the account and outbox, reconcile unknown posts/IDs, and only then restore the
affected batch to `ready`. Never blindly clear an uncertain batch. Run one
backend process; multiple publishers sharing this SQLite file are unsupported.
`/api/x-status` exposes only configuration flags and public posting history.

## Quick start

```bash
python3.13 -m venv .venv
./.venv/bin/pip install -e ".[dev]"

./.venv/bin/python scripts/build_bundled_cases.py   # build case records
./.venv/bin/python scripts/fetch_public_data.py     # refresh public data (optional)

./.venv/bin/uvicorn server.app:app --port 8000      # dashboard → http://127.0.0.1:8000
./.venv/bin/python -m plagueshield.cli assess --all --publish
```

Set the API key in `.env` at the project root (beside `pyproject.toml`):

```dotenv
OPENAI_API_KEY=your-openai-api-key
PLAGUESHIELD_LLM_MODEL=gpt-5.5
```

The server and CLI load this file automatically. Restart the backend after
editing it. Shell environment variables take precedence. `.env` is excluded
from version control, and credentials are never sent to the website.

Starting the server also starts the research worker. It assesses one bundled
case immediately, verifies guidance sources, calls GPT-5.5, and saves the result.
It then rotates through the bundled cases, waiting 120 seconds between runs.
The live page streams actual step transitions from `/api/live/stream`; hover or
keyboard focus on a step exposes its current result and elapsed time.

Worker settings in `.env`:

```dotenv
PLAGUESHIELD_WORKER_ENABLED=true
PLAGUESHIELD_RUN_INTERVAL_SECONDS=120
PLAGUESHIELD_LIVE_EVIDENCE=true
```

Use a single Uvicorn process for this embedded worker; multiple processes would
each start their own research loop. The title uses FIGlet's 3x5 ASCII lettering
with a staggered entrance and neon ripple animated by self-hosted Motion 14.0.0.
The animation pauses off the live page; reduced-motion preferences show static text.

Optional public-site configuration, also in `.env`:

```
PLAGUESHIELD_TOKEN_ADDRESS=...                # unset until the verified launch address is available
PLAGUESHIELD_TOKEN_SYMBOL=PLAGUESHIELD
PLAGUESHIELD_TOKEN_MARKET_CAP=...             # fallback if market APIs are unavailable
```

The Python code-analysis agent re-runs diagnostic scoring with each assay family
removed, then computes a posterior-odds sensitivity sweep. These experiments
audit model behavior, not clinical validity, and never execute generated code.
The GPT-5.5 analysis layer reads these experiments and the structured verdicts
before producing an evidence-grounded research brief and the final summary.
The GPT-5.5 meta-review agent then critiques all current methodology and results,
with all-history counts and detailed evidence from the latest eight backend runs.
This adds a second LLM call per iteration. It selects up to four bounded research
focus directives for the analysis prompt; immutable safety instructions and
deterministic Python methods cannot be rewritten. Code changes remain proposals
requiring human review. Revisions activate only after trusted worker publication,
starting with the next iteration, and persist in `server/prompt_revisions.sqlite3`.
Analysis and meta-review pages show prompt versions, reasons and before/after text.
These automatic changes are experimental, not evidence of improved performance;
each review includes a proposed validation plan. Public submissions cannot activate
prompt changes. Failed or invalid reviews retain the last active prompt.
Individual agent pages expose full findings, flags, calculations, linked sources,
and Python implementations. New runs also record case inputs, upstream verdicts,
execution timing, and the exact LLM request (without authentication headers).
If `OPENAI_API_KEY` is absent, the analysis agent publishes an explicit
"not configured" verdict rather than blocking local runs.

CLI:

```
plagueshield list                              # bundled case records
plagueshield assess PS-SYN-DISC-01             # one case, ASCII report
plagueshield assess --all --publish            # assess everything, post to dashboard
plagueshield assess PS-2019-MN-001 --markdown  # full auditable report
plagueshield fetch                             # refresh public data snapshots
plagueshield serve                             # run the dashboard
```

---

## Fly Deployment

Production runs at https://plagueshield.com (Fly fallback: https://plagueshield.fly.dev).
The `fly.toml` configuration runs one always-on, 1 GB shared-CPU machine in EWR.
Use exactly one machine and one Uvicorn worker: each process starts its own research
and publishing loop. Scaling requires separating the worker and moving SQLite
storage to a shared database first.

The `/data` volume retains assessments, research papers, prompt revisions and the
X publishing outbox across releases. Volume snapshots are enabled with 14-day
retention; a single volume is not high-availability storage or an independent
backup. Keep periodic off-platform exports of research data.

For the first migration, stop the local research worker and run
`python scripts/snapshot_deployment.py`. The generated `.deployment-seed` is
copied only onto a fresh volume; subsequent releases never overwrite live data.
`.env` is excluded from Docker builds. Import credentials through Fly secrets,
not build arguments or `fly.toml`. Never print or commit secret values.

Deploy updates with `flyctl deploy --remote-only --ha=false --strategy immediate`.
This single-machine strategy briefly interrupts the site while replacing the
process, but avoids concurrent autonomous workers. Check `flyctl status`,
`flyctl logs`, `/healthz` and `/api/live` after deployment.

The public production API rejects assessment submissions and manual run requests
with HTTP 403 via `PLAGUESHIELD_PUBLIC_READ_ONLY=true`; visitors can read results
but cannot trigger LLM charges or alter the audit trail. The backend worker still
publishes internally. X is disabled and in dry-run mode by default; configure
credentials as Fly secrets and verify previews before explicitly enabling it.

DNS for `plagueshield.com`: A `@` to `66.241.125.191`, AAAA `@` to
`2a09:8280:1::1a8:9991:0`; use the same addresses for `www`. Certificate requests
are registered with Fly. Run `flyctl certs check plagueshield.com` after DNS changes.

## Architecture

```
            ┌─ CASE RECORD ─ de-identified · provenance-tagged · never mutated ─┐
            └────────────────────────────┬─────────────────────────────────────┘
   stage 1  ╔═══════════════════════════ ∥ parallel ═══════════════════════════╗
            ║  DIAGNOSTIC   scores whether evidence supports Y. pestis          ║
            ║  RESISTANCE   few-shot inference over genomic + phenotypic data   ║
            ╚══════════════════════════════════════════════════════════════════╝
   stage 2  ╔═══════════════════════════ ∥ parallel ═══════════════════════════╗
            ║  EVIDENCE     retrieves current CDC/WHO guidance + literature     ║
            ║  DISCORDANCE  flags contradictions between evidence streams       ║
            ╚══════════════════════════════════════════════════════════════════╝
   stage 3  ║  UNCERTAINTY  detects OOD / weak evidence — holds the veto        ║
   stage 4  ║  NEXT-TEST    ranks by expected information gain                  ║
   stage 5  ║  ANALYSIS     GPT-5.5 synthesis of upstream verdicts              ║
   stage 6  ║  SUMMARY      renders the auditable report                        ║
                                         │
                                         ▼
                      POST /api/assessments  (append-only)
```

Agents never call each other and never mutate the case. The orchestrator owns
sequencing, so the dependency graph is declared rather than emergent, and each
agent is independently testable. A failing agent degrades the run instead of
killing it — a partial assessment that names its own gap beats a stack trace at
3am.

| Agent | Answers |
|---|---|
| **Diagnostic** | Posterior probability of plague + surveillance classification + clinical form |
| **Resistance** | Per-drug-class outlook, anomaly status, or an explicit abstention |
| **Evidence** | Which CDC/WHO guidance and literature bear on this case |
| **Discordance** | Where the evidence streams contradict each other |
| **Uncertainty** | Whether any of the above is reliable enough to report |
| **Next-Test** | Which single measurement would most reduce uncertainty now |
| **LLM Analysis** | GPT-5.5 synthesis of upstream verdicts for public status |
| **Clinical Summary** | A short, auditable report + escalation routing |

---

## What's actually novel here

### 1. Few-shot resistance prediction where positives are scarce

Resistance in *Y. pestis* is genuinely rare. A supervised classifier trained on
the public distribution learns to answer "susceptible" every time and is right
almost always while being useless. PlagueShield instead:

- carries the rarity as an **explicit Beta prior** (`fewshot/prior.py`),
  anchored to public isolate counts, with honestly wide credible intervals;
- accumulates evidence in **log-odds with named, inspectable terms** — every
  update says what moved the posterior and by how much;
- reasons by **similarity to a small bank of labelled evidence patterns**
  (`fewshot/exemplars.py`) rather than fitted weights, so the handful of
  positives are carried as named patterns instead of being averaged away.

Two consequences are reported rather than hidden: a low prior means a positive
signal needs strong evidence before it is believed, **and** a negative screen is
not reassuring in proportion — absence of evidence is weak evidence of absence
when the screen itself is incomplete.

### 2. Missingness is first-class

Every signal in the feature space carries a companion "known" channel.
"Screened and negative" and "never screened" are different points. Collapsing
them is the single most dangerous thing this system could do, because it is what
turns an untested isolate into a reassuring report. There is a test for it.

### 3. Abstention is a first-class output

`fewshot/ood.py` abstains for three independent reasons, any one sufficient:

- **No support** — unlike anything in the bank, thresholded by leave-one-out
  conformal calibration *over the bank itself*, not a hand-picked number.
- **Insufficient evidence** — nothing informative was measured. Checked
  separately, because "nothing was tested" is a well-populated region of
  feature space: a case can be perfectly in-distribution while carrying no
  information.
- **Unreliable inputs** — a negative screen on an assembly too poor to support
  it, or a single unrepeated result carrying the whole finding.

The Uncertainty Agent can only ever *lower* confidence, never raise it. Enforced
by a test.

### 4. Next test by expected information gain

For hypothesis *H* at probability *p* and a test with sensitivity *s*,
specificity *t*:

```
P(+)  = p·s + (1−p)·(1−t)
EIG   = H(p) − [ P(+)·H(p|+) + P(−)·H(p|−) ]     bits
```

EIG is then traded against turnaround, clinical actionability and feasibility
(a test needing an isolate is not available without one). Both hypotheses are
scored and weighted by which question is actually live: when plague is already
near-certain, identification information is worth little and resistance
information a great deal.

One subtlety the implementation handles explicitly: **when the Resistance Agent
abstains, its posterior is not a calibrated belief** — it is the base rate,
barely moved. Feeding that into EIG yields near-zero entropy and the conclusion
that susceptibility testing is not worth doing, which is exactly backwards. EIG
is therefore evaluated at a probability pulled toward maximum entropy in
proportion to acknowledged ignorance. With complete evidence it is a no-op.

### 5. Correlated evidence is damped

A naive product of likelihood ratios assumes conditional independence. That
assumption fails here in the dangerous direction — toward false certainty. PCR,
culture and antigen detection on the *same bubo aspirate* are three views of one
sample, not three independent observations; "endemic area" + "outbreak contact"
+ "part of a cluster" is largely one fact.

Evidence is grouped and combined with geometric damping — strongest term in
full, next at *d*, next at *d²* — with positive and negative terms damped
separately so contradicting evidence is never suppressed for arriving second.
The first implementation returned 100.0% on five of eight cases and 93.6% on a
*single field rapid test*; after damping the latter reads 75%. There is a
regression test for it, and a hard ceiling below 1.0 because no finite evidence
justifies certainty.

---

## Scope boundary on resistance

**PlagueShield contains no catalogue of resistance mechanisms**, and
`knowledge/regimens.py` says so at the top of the file so nobody "improves" it
back in. Specifically absent: resistance genes, alleles, mutations or plasmids;
any mapping from a determinant to the resistance it confers; any ranking of
which determinants would defeat which plague regimens.

That catalogue is the one artifact here that would function as a design sheet
for an untreatable strain, and it is not needed. Determinant calling is
delegated to externally maintained tools (NCBI AMRFinderPlus, ResFinder,
CARD/RGI) which emit a determinant plus its drug class. PlagueShield consumes
the **class** and reasons from there.

This is not a capability sacrifice. The question an ID physician actually asks —
*"is a class in my regimen under threat, and does the phenotype agree?"* — is
fully answerable at class level, and class level is also where guidance is
written. It is also how real pipelines are built.

---

## Public data

What is and is not available, stated plainly because it shapes the system:

| | Availability | Used for |
|---|---|---|
| **Isolate genomes + AMR genotype calls** | Genuinely public (NCBI) | Calibrating the resistance prior |
| **Outbreak reporting** | Public in aggregate (WHO DON, CDC) | Epidemiological context, endemic geography |
| **Individual patient records** | **Not public, and should not be** | — |

Plague case counts are small enough that a real line list would be
re-identifiable. Anyone offering one is either synthesising it or leaking it.
PlagueShield therefore reconstructs case *vignettes* and labels every record
with its origin:

- `published_report` — reconstructed from a public outbreak report; detail is
  limited to what the source stated, anything else left null rather than invented
- `surveillance_summary` — representative case from aggregate surveillance;
  a pattern, not a person
- `synthetic` — built to exercise a specific behaviour; no epidemiological meaning

`provenance` is a required field, is shown in the UI, and feeds a reliability
penalty in the Uncertainty Agent.

**Live sources** (`scripts/fetch_public_data.py`, all keyless):

- NCBI E-utilities — *Y. pestis* BioSample metadata and public assembly counts
  (3,670 as of 2026-10-05)
- WHO Disease Outbreak News OData API — paged; plague DONs are rare enough that
  a single page contains none, so it scans ~800 items to find them

All fetchers degrade gracefully: no network yields a recorded reason, never an
exception that stops a clinical assessment.

### Bundled cases

| Case | Origin | Exercises |
|---|---|---|
| `PS-2019-MN-001` | published report | Marmot exposure, bubonic→pneumonic, full workup |
| `PS-2017-MG-014` | published report | Urban pneumonic; antibiotics before culture |
| `PS-2021-US-003` | surveillance | US bubonic, complete reference-lab characterisation |
| `PS-2020-CD-021` | surveillance | Remote health post — single field RDT, sparse |
| `PS-2022-US-009` | surveillance | Septicaemic, partial panel, MALDI misidentification |
| `PS-SYN-DISC-01` | **synthetic** | Genome clean, phenotype non-susceptible → critical conflict |
| `PS-SYN-OOD-01` | **synthetic** | Minimal evidence → explicit refusal to predict |
| `PS-SYN-SPEC-01` | **synthetic** | Sequencing contradicts plague-targeted assays |

---

## Example output

```
══════════════════════════════════════════════════════════════════
  PLAGUESHIELD :: ASSESSMENT REPORT
══════════════════════════════════════════════════════════════════
  CASE                              PS-SYN-DISC-01
  ORIGIN                            SYNTHETIC
──────────────────────────────────────────────────────────────────
  PLAGUE LIKELIHOOD                 VERY HIGH
  STANDARD-TREATMENT SUSCEPTIBILITY POSSIBLY COMPROMISED
  RESISTANCE ANOMALY                SUSPECTED
  CONFIDENCE                        MODERATE
  MOST VALUABLE NEXT RESULT         REPEAT SUSCEPTIBILITY TESTING
                                    ON FRESH SUBCULTURE (REFERENCE
                                    LABORATORY)
──────────────────────────────────────────────────────────────────
  ESCALATION
    > HUMAN REVIEW REQUIRED — genomic and phenotypic results
      disagree. Phenotype governs interim management; the conflict
      must be adjudicated by a clinical microbiologist.
    > Resistance anomaly in a core therapeutic class — notify the
      reference laboratory and the responsible public health
      authority. Resistance in Y. pestis is rare and reportable.
──────────────────────────────────────────────────────────────────
  HUMAN REVIEW: REQUIRED
══════════════════════════════════════════════════════════════════
  Decision support only. Does not prescribe treatment.
══════════════════════════════════════════════════════════════════
```

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/assessments` | **The agent posts results here** |
| `GET` | `/api/assessments` | Latest assessment per case |
| `GET` | `/api/assessments/{id}` | Full assessment (`?version=n` for history) |
| `GET` | `/api/assessments/{id}/report` | `?fmt=markdown\|ascii` |
| `GET` | `/api/cases` · `/api/cases/{id}` | Bundled case records |
| `POST` | `/api/run/{id}` · `/api/run-all` | Run the pipeline |
| `GET` | `/api/public-data` | Public source snapshots |
| `GET` | `/api/agents` · `/api/stats` | Pipeline shape, dashboard counters |

The store is **append-only** NDJSON. Re-assessing a case adds a version rather
than replacing one, so a reviewer can see how the picture changed as results
arrived — an audit trail that can be silently rewritten is not an audit trail.

---

## Tests

```bash
OPENAI_API_KEY='' PLAGUESHIELD_WORKER_ENABLED=false ./.venv/bin/python -m pytest tests/ -q
```

Tests assert *properties*, not numbers. A test pinning the posterior to three
decimals breaks on every calibration change and tells you nothing; a test
asserting "never reports certainty" catches a real regression. Covered:

- never reports certainty; damping discounts correlated evidence but does not
  suppress contradicting evidence
- a single field RDT is not near-certain (regression guard)
- probability and case classification stay independent
- **absent evidence is never reported as susceptible**
- an untested isolate abstains rather than reassures
- a poor-quality genome does not license a clean screen
- genotype/phenotype conflict is critical and escalates
- uncertainty can only lower confidence, never raise it
- EIG is zero at certainty, peaks at maximum uncertainty, never exceeds prior
  entropy, and is zero for a coin-flip test
- a failing agent degrades rather than crashes
- no report ever issues dosing instructions

---

## Layout

```
plagueshield/
  models.py              domain models; provenance required throughout
  orchestrator.py        staged pipeline + dashboard publisher
  agents/                the ten agents
  knowledge/
    assay_performance.py published sens/spec + likelihood ratios, cited
    regimens.py          drug-class → treatment role (scope boundary documented)
    guidance.py          offline-first CDC/WHO corpus
  fewshot/
    prior.py             Beta prior + log-odds evidence accumulation
    exemplars.py         labelled evidence patterns + similarity vote
    ood.py               conformal abstention
  data/
    cases/               bundled de-identified records
    public_sources.py    NCBI + WHO fetchers
server/
  app.py                 FastAPI; append-only store
  worker.py              background runs and per-agent progress
  static/                dashboard
```
