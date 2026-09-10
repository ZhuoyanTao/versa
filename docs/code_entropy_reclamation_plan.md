# Code Entropy Reclamation Plan

Locked: 2026-09-09  
Audit baseline: `37a8dfa`  
Status: all five implementation items completed; see the completion record below.

## Objective and scope

Reduce maintenance surface without removing supported metrics, changing scoring
results, or breaking existing entrypoints, aliases, and output formats. The audit
surveyed 317 tracked files, including structural analysis of all 156 tracked
Python files, consumer searches, runtime tracing, and relevant Git history.

This document records the audit's five ranked candidates and their acceptance
criteria. The original plan was accepted before implementation. Each implementation batch
must recheck consumers and the current checkout before changing code. Estimated
line reductions are planning estimates after replacement glue, not targets or
proof of safety.

## Implementation order

| Order | Candidate | Confidence | Risk | Estimated net reduction |
| --- | --- | --- | --- | --- |
| 1 | Shared Qwen postprocessing rules | High | Low–medium | 500–600 lines |
| 2 | Shared deterministic test waveform generator | High | Low | 400–430 lines |
| 3 | Remove fallback YAML implementation | High | Low | About 40 lines |
| 4 | Shared discovery/runtime metadata definitions | High in duplication | Medium | 80–140 lines initially |
| 5 | Shared scorer orchestration | Medium | Medium | 100–180 lines |

Keep each item independently reviewable. Do not combine these into a registry or
framework rewrite. Narrow a candidate or defer it if new evidence shows that it
would change a supported contract or merely move complexity elsewhere.

## 1. Consolidate Qwen postprocessing rules

### Evidence

- `scripts/postprocess/qwen2_audio_json_output_standardizer.py` and
  `scripts/postprocess/qwen2_audio_jsonl_standardizer_batch.py` contain identical
  `EXPECTED_FORMATS` and `TRANSLATION_DICT` dictionaries.
- AST comparison found identical bodies for `_translate_non_english`,
  `_process_llm_output`, and `_rules_based_standardize`.
- `scripts/postprocess/check_llm_result_match.py` repeats the format schema but
  omits `qwen_overlapping_speech`; its other entries match.
- These are executable scripts. `scripts/postprocess/post_process.sh` invokes
  the JSONL path. Different file and inference modes remain real capabilities.

### Planned cut

Extract one shared schema, translation dictionary, and set of pure normalization
functions. Preserve the JSON and JSONL entrypoints, file handling, and separate
inference paths. Preserve the filter's existing subset explicitly; accepting
`qwen_overlapping_speech` there is a separate behavior decision.

Do not merge model initialization or batching merely because normalization code
is shared. Keep the extraction independent of the proposed Prompt Bank feature.

### Acceptance and verification

- No intended observable behavior change.
- Compare old and new outputs for all categories, numeric fields, translations,
  malformed responses, and unknown keys with model calls mocked.
- Verify both direct script entrypoints can import the shared implementation.
- Verify the filter still preserves its existing accepted-key subset.
- Recheck consumers and search for stale duplicate definitions after extraction.

## 2. Share the deterministic test waveform generator

### Evidence

Fifteen metric test modules contain an identical 32-line `generate_fixed_wav`
implementation. These generate inputs rather than independently implement the
algorithm under test:

- `test/test_metrics/test_asr_matching.py`
- `test/test_metrics/test_asvspoof.py`
- `test/test_metrics/test_audiobox_aesthetics.py`
- `test/test_metrics/test_cdpam_distance.py`
- `test/test_metrics/test_chroma_alignment.py`
- `test/test_metrics/test_discrete_speech.py`
- `test/test_metrics/test_dpam_distance.py`
- `test/test_metrics/test_emo_similarity.py`
- `test/test_metrics/test_emo_vad.py`
- `test/test_metrics/test_nisqa.py`
- `test/test_metrics/test_nomad.py`
- `test/test_metrics/test_noresqa.py`
- `test/test_metrics/test_owsm_lid.py`
- `test/test_metrics/test_pam.py`
- `test/test_metrics/test_pesq_score.py`

### Planned cut

Keep one test utility or fixture exposing the existing parameters. Preserve each
metric's assertions, fixture ownership, and distinct prediction/reference inputs.
Choose a location that works with the repository's pytest import convention.

### Acceptance and verification

- Generated WAV bytes remain identical for parameter combinations used by callers.
- Affected tests collect and run with `--import-mode=importlib`.
- Preserve optional dependency and real-model skip gates; record unavailable
  checks rather than enabling downloads to validate a test-only extraction.
- No production API or metric behavior changes.

## 3. Remove the fallback YAML implementation

### Evidence

`versa/metric_discovery.py` makes PyYAML optional and implements `_safe_dump_yaml`,
`_dump_yaml_item`, `_dump_yaml_dict`, and `_dump_yaml_scalar` to compensate.
`pyproject.toml` declares PyYAML as a required base dependency. The fallback arrived
with discovery in `4e49097`; the audit found no documented requirement for
recommendations to work without declared base dependencies.

### Planned cut

Import PyYAML normally and use `yaml.safe_dump(config, sort_keys=False)` directly.
Delete the fallback branch and four serialization helpers.

### Acceptance and verification

- Generate recommendations for every supported task/device pair.
- Parse each recommendation with PyYAML and compare its configuration values,
  including nested predictor arguments.
- Preserve recommendation headers and output from the existing PyYAML path.
- Explicit tradeoff: recommendation output requires the declared base
  dependencies to be installed. Revisit the cut if a supported dependency-free
  contract is discovered.

## 4. Give discovery and runtime shared metadata definitions

### Evidence

`versa/metric_discovery.py` manually recreates metadata for Qwen2-Audio, Qwen-Omni,
SQUIM, and ScoreQ. Runtime metadata helpers repeat those facts in:

- `versa/utterance_metrics/qwen2_audio.py`
- `versa/utterance_metrics/qwen_omni.py`
- `versa/utterance_metrics/squim.py`
- `versa/utterance_metrics/scoreq.py`

The duplication serves a real requirement: discovery must remain usable without
loading optional model stacks. Deleting source discovery outright would not
preserve that requirement.

### Planned cut

Move the duplicated definitions into dependency-light shared definitions used by
both runtime registration and discovery. Start with these four families. Remove
their redundant metadata construction only after both consumers use the shared
source. Do not undertake a wholesale registry rewrite or externalize legacy
prompts as part of this cut.

### Acceptance and verification

- Compare every metadata field and alias between discovery and runtime.
- Preserve generated Qwen names, legacy aliases, and default prompt behavior.
- Test discovery in fresh processes without optional model dependencies.
- Confirm discovery does not initialize or download models.
- Verify shared definitions are included in the installed package.
- Net reduction must include any new integration glue; defer if the new mechanism
  is as complex as the duplication it replaces.

## 5. Consolidate scorer orchestration

### Evidence

`versa/bin/scorer.py` and `versa/bin/scorer_chunk.py` repeat device selection,
logging, input loading, configuration handling, and scoring orchestration. They
have diverged: the chunked scorer forwards `cache_folder`, while the main scorer
only parses it; metric-oriented scoring exists only in the main scorer.

### Planned cut

Share orchestration after input preparation while preserving both entrypoints and
chunk generation. Migrate internal uses of legacy wrappers where appropriate,
but retain public compatibility paths unless external obligations are resolved.

Before implementation, record the two entrypoints' current behavior in a compact
comparison. Preserve their differences explicitly where needed. Fixing cache
forwarding or changing corpus input selection is a separate behavior change,
not an implicit consequence of extraction.

### Acceptance and verification

- Exercise ordinary and chunked entrypoints with mocked metrics.
- Cover corpus input selection, cache overrides, reference/text handling, chunk
  identifiers, resume, and the existing metric-oriented mode.
- Preserve JSONL and corpus output contracts and existing entrypoint arguments.
- Preserve model release and file-handle cleanup ownership.
- Keep validation at audio, configuration, and persisted-result boundaries.

## Explicitly deferred or protected surfaces

- Public `GPUMetric`, preprocessing hooks, and `create_metric_suite`: sparse
  repository consumption does not establish absence of external consumers.
- `MetricSuite.compute_parallel`: currently an explicit unsupported-operation
  contract, reinforced by a deliberate history change and a test. Do not delete
  it solely because it is unimplemented.
- Vendored model code, distinct ASR backends, checkpoint compatibility, and
  resource cleanup: duplication alone does not justify removal.
- Existing YAML aliases, output keys, persisted formats, and user capabilities:
  changes require an explicit compatibility or product decision.
- Untracked files and local environments are user work, not deletion candidates.

At audit time, untracked work included `.codex-test-venv/`, `tools/nomad/`,
`docs/development_roadmap_2026-09.md`, and
`docs/prompt_bank_implementation_plan.md`. The roadmap describes a lazy registry
module absent from the audited checkout. Reconcile such claims against actual
code before using them as implementation evidence; do not rewrite those plans as
part of entropy reclamation.

## Validation baseline and completion rules

The audit ran:

```bash
PYTHONDONTWRITEBYTECODE=1 .codex-test-venv/bin/python -m pytest \
  -q -p no:cacheprovider test/test_metrics/test_definition.py
```

Result: **11 passed, 9 dependency warnings**. The full model-backed suite was not
run. No tracked source files were changed by the audit.

For each implementation batch:

1. Read current repository instructions, inspect Git status, and recheck consumers.
2. Establish a proportional baseline for the affected behavior.
3. Run the candidate's decisive checks, then relevant core/pipeline checks and
   repository formatting/lint gates. Record skips and pre-existing failures.
4. Search for stale symbols, imports, schemas, and documentation.
5. Run `git diff --check` and inspect the complete diff for scope expansion.
6. Record actual files/lines/concepts removed, behavior changes if any, exact
   validation results, and compatibility obligations intentionally retained.

A batch is complete only when it reduces maintenance surface and preserves the
agreed observable contracts. Green tests and negative line counts alone are not
sufficient evidence.


## Completion record — 2026-09-10

Implemented all five items. The branch started at the audit baseline, then was
rebased onto upstream `caf9cbe` so the PR preserves features merged since the
audit. The other untracked plans, local environment, and `tools/nomad/` were not
included or edited.

### Completed cuts and retained boundaries

- [x] Qwen schema, translations, and three pure normalization functions now live
  in `scripts/postprocess/qwen_normalization.py`. Both standardizer classes retain
  their methods, constructors, file modes, and separate inference implementations.
  The filter derives its original subset by excluding overlapping speech.
- [x] The fifteen listed metric modules use `test/audio_utils.py`. Their fixtures,
  assertions, optional dependency gates, and distinct inputs remain local.
  Chroma's old helper had different defaults, but all its callers supply those
  parameters explicitly. All 33 existing calls produce byte-identical WAVs.
- [x] Recommendation serialization uses the required PyYAML dependency directly.
  The four fallback helpers are removed. Missing required base dependencies are
  not supported; no other dependency or output contract changes.
- [x] Qwen2-Audio, Qwen-Omni, SQUIM, and ScoreQ metadata comes from
  `versa/metric_metadata.py`. Runtime helper names remain importable from their
  original modules. Qwen aliases share definitions; the other aliases remain
  discoverable from their original registration calls. Prompts stay in place.
- [x] `versa/bin/scoring.py` owns shared device/logging setup, input loading,
  configuration/cache loading, and ordinary/metric/corpus orchestration. The
  entrypoints retain early config validation, multi-source handling, reporting,
  and chunk preparation. Legacy scorer wrappers and cleanup ownership remain.

### Scorer contracts reconciled before extraction

| Behavior | Ordinary CLI | Chunk CLI |
| --- | --- | --- |
| Corpus input | Loaded file mappings | Original paths; chunk prediction directory and no reference when chunking |
| Corpus `io` default | No CLI override | Defaults to CLI `io`; explicit config wins |
| Corpus reference eligibility | Loaded reference mapping exists | Reference argument exists, even with `--no_match` |
| Reference count precheck | Required | Bypassed while chunking |
| Metric loop / local workers / reports / multi-source | Existing options retained | Existing utterance loop retained |
| Output / resume | Existing JSONL, corpus YAML, and resume behavior | Same formats; existing chunk keys and replicated text |

At audit baseline `37a8dfa`, only the chunk CLI forwarded the corpus cache
argument. Upstream subsequently replaced this divergence with shared cache
configuration in both CLIs. The final extraction preserves that upstream policy,
including explicit per-metric overrides, rather than restoring the obsolete
asymmetry. Upstream also added early config validation: empty configuration now
fails through the parser in both entrypoints, as it does on the PR target.

### Verification evidence

All pytest runs used `.codex-test-venv/bin/python`,
`PYTHONDONTWRITEBYTECODE=1`, `-q -p no:cacheprovider`, and
`--import-mode=importlib` for repository tests.

- Original baseline: `test/test_metrics/test_definition.py test/test_general.py`
  — **14 passed**.
- Differential execution against the original source: **3,508 normalization
  comparisons**, **33 WAV byte comparisons**, every metadata field/alias for all
  four families, and all eight task/device recommendation outputs matched.
  Cases included every category, numeric bounds, translation ordering, malformed
  strings, unknown keys, and the filter subset. Historical partial translations
  were preserved rather than silently corrected.
- Final core/pipeline suite: `test/test_qwen_postprocess.py`,
  `test/test_metrics/{test_shared_metadata,test_definition,test_distributional_metrics}.py`,
  `test/test_pipeline/{test_scorer_entrypoints,test_local_workers,test_mapss_pipeline}.py`,
  `test/test_reporting.py`, and `test/test_general.py` — **79 passed**.
- Entrypoint checks were then extended with absent text / literal `None` reference
  cases; the final focused suite passed **19 tests**. It covers corpus inputs,
  cache overrides, chunk keys/text, resume, model release, file closure, reports,
  worker forwarding, and validation before audio loading. Sixteen applicable
  contract cases also passed against untouched upstream scorer entrypoints.
- All fifteen affected metric modules ran offline with `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1`: **101 passed, 4 skipped, 21 failed**; all **126 tests collect**. Seventeen failures
  require uncached Hugging Face models (six DiscreteSpeech, six EmoVad, five PAM);
  four NORESQA failures are a local `fairseq.meters` checkpoint-import issue.
  All twenty-one failed cases reproduced against untouched upstream `caf9cbe`
  with matching import/cache paths (21 failed, 8 passed, 2 skipped in the four
  affected modules). The original pre-rebase run had 107 passed, 4 skipped, and
  12 failures, also reproduced on the audit baseline. Optional gates were not
  weakened and no model downloads were enabled.
- Direct standardizer script `--help` checks pass from outside the repository.
  A fresh-process discovery test rejects any import of torch, torchaudio,
  transformers, librosa, or ScoreQ, and passes.
- A wheel built with `pip wheel --no-deps --no-build-isolation`; its contents include
  the shared metadata module, and discovery from the extracted wheel passes.
- Black passes on all **177 tracked Python files**; the repository's blocking
  Flake8 gate (`--select=E9,F63,F7,F82`) reports **0 findings**. New files also pass
  full Flake8. Across the
  changed files Flake8 reports **67 pre-existing findings versus 73 at baseline**,
  with no new findings. `git diff --check` and stale-definition searches pass.

No supported metric, alias, prompt, public compatibility wrapper, persisted
format, or model cleanup mechanism was removed. The net change is **718 fewer Python lines**,
including the added regression tests: normalization -532; test waveform helpers
-485; metadata/YAML -110; scorer orchestration -61; regression coverage +470. The accepted plan is tracked here for review, so its
previously untracked prose is additional documentation in the PR.
