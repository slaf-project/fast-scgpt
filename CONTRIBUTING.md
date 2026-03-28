# Contributing

Thanks for your interest in fast-scGPT. This repo is a **reference implementation** used alongside [SLAF](https://github.com/slaf-project/slaf) and Modal; it is not yet optimized as a library for arbitrary reuse.

## Before you start

- Open an issue (or discussion on your usual channel) for **large changes** so we can align on direction.
- Match **existing style**: run `ruff check`, `ruff format`, and `mypy` on `fast_scgpt/` the way CI / pre-commit does (`uv run pre-commit run --all-files` if you use pre-commit).

## Pull requests

- Keep diffs **focused** on one concern.
- Add or update **tests** in `tests/` when behavior changes.
- Update **README.md** if you change how people install or run training.

## License

By contributing, you agree that your contributions will be licensed under the same terms as this project ([LICENSE.md](LICENSE.md)).
