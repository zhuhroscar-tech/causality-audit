# Changelog

## v0.2.5 — Package resource links and release-tag CI

- Added package metadata links for the issue tracker and changelog alongside the homepage.
- Made release-tag CI coverage explicit for `v*` tags.
- Added repository-contract checks so package resource links and release-tag CI wiring stay in sync.

## v0.2.4 — Repository completeness contracts

- Added release-history documentation and repository-contract tests so packaging, CI, changelog, and release artifact expectations stay explicit.
- Linked release history from the English and Chinese READMEs.

## v0.2.3 — Packaging metadata fix

- Modernized package license metadata to current SPDX form.
- Declared packaged license files explicitly and removed deprecated license classifier metadata.
- Raised the setuptools build backend floor to a version that supports the SPDX license metadata.

## v0.2.2 — NaN leak detection fix

- Fixed a silent-clean edge case where non-finite layer output differences could be misreported instead of treated as suspicious audit output.

## v0.2.1 — CLI robustness fix

- Hardened demo argument validation so invalid sequence and chunk sizes fail cleanly during argument parsing.

## v0.2.0 — Shape-mismatch handling

- Reported data-dependent layer output shape mismatches as `inconclusive` audit results instead of crashing or silently comparing incompatible outputs.

## v0.1.0 — Prefix-invariance audit

- Initial release of the NumPy two-forward-pass prefix-invariance audit.
- Added synthetic clean and buggy chunked-scan reference models for validating audit harnesses.
