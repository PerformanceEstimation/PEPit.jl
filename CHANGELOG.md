# Changelog

All notable changes to `PEPit.jl` are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-06-08

### Added
- Documenter.jl documentation scaffold with API reference pages, an executable
  Quick Start, tutorials, contributing guide, release notes, and custom styling.
- GitHub Actions workflow for building and deploying the documentation site.
- Comprehensive source docstrings for the public API and key implementation
  helpers across the core objects, function classes, operator classes,
  primitive steps, and utilities.
- Mathematical docstrings for all `wc_*` example functions.
- Executable, expanded Quick Start covering worst-case instance recovery with
  `evaluate`, the explicit dual certificate (`solve_dual!`, `eval_dual`,
  `DualPEPCertificate`), and dimension-reduction heuristics (`tracetrick`,
  `logdetiters`).
- Contributing guide with templates for adding function/operator classes,
  primitive steps, and worked examples.
- Automatically generated Examples overview, category pages, and per-example
  pages extracted from the corresponding `wc_*` function docstrings.
- This changelog and a "Release notes" page in the documentation.

### Changed
- `evaluate` is now exported. It is the documented way to recover numerical
  realizations of `Point`/`Expression` objects after solving.
- Standardized all documentation links and documentation deployment on
  `github.com/PerformanceEstimation/PEPit.jl`.
- Tests now access internal counters and helper functions as `PEPit.<name>`
  after those symbols were removed from the public export list.

### Fixed
- `evaluate` was documented and shown in the Quick Start but not exported, so
  user code following the Quick Start raised `UndefVarError`.
- Broken display-math block in the `inexact_gradient_step!` docstring: the
  conditional definition is now contained in a single `math` fence (previously
  part of it rendered as raw LaTeX / a code block).
- De-Pythonized the `inexact_gradient_step!` docstring: a `# Throws` section now
  documents the real `ErrorException`, with Julia string/list syntax.

### Removed
- Internal symbols are no longer exported into the public namespace: the global
  counters (`Point_counter`, `Expression_counter`, `Function_counter`,
  `Global_Constraint_counter`, `PSDMatrix_counter`, `NEXT_ID`) and the internal
  helpers (`_is_already_evaluated_on_point`,
  `_separate_leaf_functions_regarding_their_need_on_point`,
  `_get_nb_eigs_and_corrected`). They remain reachable as `PEPit.<name>`.

## [0.1.1]

- Initial documented release: the core PEP workflow (`PEP`, `Point`,
  `Expression`, `Constraint`, `PSDMatrix`), function and operator interpolation
  classes, primitive steps, `solve!` / `solve_dual!`, and ~96 worked examples
  validated against the Python `PEPit` package.
