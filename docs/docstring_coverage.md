# Docstring coverage audit

Audited 2026-09-11 against checkout `bf8804f` plus this documentation change.

## Recent automated review findings

The warnings are from **CodeRabbit**, not pytest coverage or a locally configured
Python linter. The latest walkthrough comments retrieved during this audit say:

| PR | Reported coverage | Analyzed scope | Comment updated (UTC) |
| --- | ---: | --- | --- |
| [#96: Reclaim code entropy](https://github.com/wavlab-speech/versa/pull/96#issuecomment-5615048983) | 26.39% | 72 touched functions, 33 files; one unsupported file skipped | 2026-09-10 08:00 |
| [#93: MAPSS integration](https://github.com/wavlab-speech/versa/pull/93#issuecomment-5593556287) | 24.39% | 41 touched functions, 9 files | 2026-09-09 10:53 |
| [#95: Multi-source signal metrics](https://github.com/wavlab-speech/versa/pull/95#issuecomment-5609504597) | 68.42% | 19 touched functions, 2 files | 2026-09-09 23:56 |

All require 80%. The current #93 comment supersedes an earlier recorded
24.32% / 37-function result. These are mutable comments: their update timestamps
are not proof of the exact source revision evaluated. They do not publish a
checker version, evaluation base/head pair, or function-by-function inventory.
The local measurements below therefore do not reproduce or replace these bot
results. A new CodeRabbit review on a published head is needed to verify its
current diff-specific gate; no review was requested or PR posted in this audit.

CodeRabbit documents its check as applying to functions touched by the diff.
This explains why documentation of test helpers and script adapters matters in
addition to library APIs. [CodeRabbit pre-merge checks](https://docs.coderabbit.ai/pr-reviews/pre-merge-checks)

PR #95's multi-source changes are not in this checkout. The existing
`signal_metric` documentation explicitly describes its scalar, single-source
limitation rather than promising that PR's permutation support.

## Reproduced local results

Both third-party tools were run from an isolated environment with pinned versions.
The baseline was reconstructed from tracked Python files at `bf8804f`.

| Check | Baseline | Updated | Threshold |
| --- | ---: | ---: | ---: |
| Interrogate 1.7.0, full package | 461 / 1024 = 45.0% | 854 / 1024 = 83.4% | 80% |
| docstr-coverage 2.3.2, full package | 461 / 1019 = 45.2% | 854 / 1023 = 83.5% | 80% |
| Python AST, functions only | 301 / 792 = 38.01% | 636 / 792 = 80.30% | 80% |

Scope is **all 78 Python files under `versa/`**, including the bundled NISQA,
NORESQA, and PAM implementations. No paths, private functions, constructors,
nested functions, or magic methods were excluded to reach the target. Setter
and deleter checks are explicitly enabled for docstr-coverage. Tests and scripts
are outside these package totals, but the missing function docstrings in files
associated with #93 and #96 were also addressed.

For an additional local inventory, the current versions of the 33 Python files
changed between `caf9cbe` and `bf8804f` (#96) have **328/328** documented functions.
The eight Python files changed by merge `caf9cbe` relative to its first parent
(#93) have **122/122**. These count every current function in those files, not
only changed functions; deleted definitions are absent. They are deliberately
not labeled as reproductions of CodeRabbit's 72- and 41-function denominators.

The package gained 393 literal docstrings, including 335 function docstrings.
The differing totals come from checker semantics: Interrogate counts empty
modules, while docstr-coverage omits them. Four previously empty package
initializers now contain documentation, increasing the latter denominator.

The third-party measurements used Python 3.14; regression tests used Python 3.12.
CI runs the static checks on Python 3.10 without installing model dependencies.

## What to document

Prioritize the contract a caller cannot infer from a function name:

- **Metric compute methods:** waveform axes, channel selection/mixing, sample-rate
  defaults, alignment, required references/text, output keys and units, numerical
  limitations, and errors. Examples: MAPSS source order and diagnostic artifacts;
  LogWMSE's configured model rate; WER edit counts versus normalized error rates.
- **Setup and cache helpers:** optional dependencies, downloads, device selection,
  global cache changes, and local/offline asset behavior.
- **Scoring and reporting:** file ownership, append/resume behavior, dispatch,
  missing/nonfinite values, score-direction heuristics, and report overwrites.
- **Registry/discovery:** metadata extraction without model imports, aliases,
  configuration inspection, and lazy runtime loading.
- **Tests and adapters:** the invariant asserted, fake backend behavior, or the
  compatibility contract being preserved. Short summaries are sufficient here.

The remaining missing functions are in bundled model internals; the AST check
prints their exact file, line, and qualified name. Coverage establishes presence,
not correctness or completeness. Review the documentation alongside the code.

## Checker survey and selection

| Source | Useful for | Limitation / decision |
| --- | --- | --- |
| [Interrogate](https://interrogate.readthedocs.io/en/latest/) | Static package inventory, detailed missing-definition reports, percentage gate | Selected at 1.7.0 with default separate class/constructor counting and no ignore flags. |
| [docstr-coverage](https://github.com/HunterMcGushion/docstr_coverage) | Independent package inventory and percentage gate | Selected at 2.3.2 with setters/deleters included; empty modules differ from Interrogate. |
| [Ruff D103 / pydocstyle rules](https://docs.astral.sh/ruff/rules/undocumented-public-function/) | Missing public-function documentation and docstring style rules | Useful future style linting; not an interchangeable package coverage percentage. |
| Python `ast` inventory in `ci/check_function_docstrings.py` | Explicit count of actual function/async-function nodes | Selected as the function-only gate; counts literal, nonempty docstrings, including nested/private definitions. Does not infer inherited documentation. |
| CodeRabbit | Existing PR changed-function gate | Keep separate; its published comments do not expose an exact local reproduction recipe. |

A relevant 2.3.2 implementation quirk: docstr-coverage's `--skip-class-def` uses
name capitalization rather than the AST node type, and `--skip-file-doc` can
still count documented modules. Those options changed the denominator during
this audit and were **not** used as a true function-only gate. The small AST
counter avoids that ambiguity. See the pinned implementation's
[node filtering](https://github.com/HunterMcGushion/docstr_coverage/blob/v2.3.2/docstr_coverage/coverage.py)
and module collection logic.

## Repeat the checks

Run from the repository root in a Python environment:

```bash
python -m pip install -r ci/requirements-docstrings.txt
interrogate --verbose --verbose --fail-under 80 versa
docstr-coverage --include-setter --include-deleter --fail-under 80 versa
python ci/check_function_docstrings.py --fail-under 80 versa
```

All three commands fail below 80%. The workflow runs both cross-checks even if
the first check fails, so their missing-definition reports remain visible.
The AST counter also accepts individual Python files or directories for a
focused inventory, but it is not a reproduction of CodeRabbit's diff selection.

Validation: 118 tests passed across the regression selection and counter/cache
checks, with one optional real-model test skipped. The new counter has tests for
nested/async/private/init counting, exact threshold behavior, and empty-input
rejection. All 177 tracked Python files passed Black; blocking Flake8 checks
and `git diff --check` passed. Executable ASTs of all 74 existing modified Python
files were compared with HEAD after removing only literal docstrings; scoring,
signatures, and test behavior are unchanged.
