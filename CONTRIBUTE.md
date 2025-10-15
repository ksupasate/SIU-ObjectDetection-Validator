# Contributing Guidelines

Thanks for your interest in improving the SIU Object Detection Validator! This document outlines how to propose changes and keep the project healthy.

## Getting Started

- **Discuss first:** Open an issue for new features or substantial bug fixes so we can align on scope before you invest time.
- **Set up environment:** Follow the steps in `README.md` to install dependencies and download the dataset.
- **Branching:** Work on a feature branch named `feature/<short-description>` or `fix/<short-description>`.

## Code Standards

- **Style:** Follow PEP 8 for Python code unless project conventions dictate otherwise. Keep imports ordered and remove unused ones.
- **Type hints:** Maintain existing type annotations and add new ones where they improve clarity.
- **Logging:** Use the structured logger (`logging.getLogger("SIU.*")`) instead of `print` statements within the library.
- **Comments:** Add concise comments only when the intent is non-obvious.

## Testing & Validation

- **Unit/Integration tests:** Add tests when you introduce new behavior or fix bugs that previously slipped through.
- **Manual checks:** Run `python main.py train` and/or `python main.py inference <image>` when touching training or inference logic (attach logs or describe manual validation in the pull request).
- **Static checks:** Ensure the code passes `python -m compileall` or your preferred linting setup if you introduce new modules.

## Documentation

- Update `README.md`, `QUICKSTART.md`, or other docs when user-facing behavior changes.
- Keep changelog entries in your pull request description if the change is noteworthy.
- Screenshots or metrics are welcome when they demonstrate improvements.

## Pull Request Checklist

Before submitting a PR, confirm that:

1. The branch is rebased on the latest `main`.
2. All new files include appropriate licenses or headers if required.
3. Tests (automated and/or manual) are described and passing.
4. Documentation updates accompany user-visible changes.
5. You filled out the PR template (if available) with context, testing, and reviewer guidance.

## Code Review Expectations

- Be responsive to feedback and happy to iterate.
- Keep commits focused; prefer small, self-contained commits over large mixed changes.
- If you disagree with feedback, start a discussion—shared context makes for better decisions.

We appreciate every contribution, from typo fixes to large features. Thank you for helping improve the SIU Object Detection Validator!
