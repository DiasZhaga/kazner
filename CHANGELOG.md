# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Package skeleton: `pyproject.toml` (src layout, pinned runtime dependencies, `dev` extra),
  `kazner` console script stub with `--version`, ruff and pytest configuration, MIT licence,
  README skeleton and documentation stubs (#1).
- GitHub Actions CI: ruff lint and format check, offline tests with coverage on Ubuntu and
  Windows with CPU-only PyTorch, network/smoke job with a cached Hugging Face hub; CI badge (#2).
