# Contributing to bvar

## 1. Set up your fork

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-github-user>/bvar.git
cd bvar
```

Git names your fork `origin`; this command names the main repository `upstream`, so you can fetch updates from it:

```bash
git remote add upstream https://github.com/bank-of-england/bvar.git
```

Install the package with development dependencies in editable mode:

```bash
pip install -e ".[dev]"
```

Install the pre-commit hooks:

```bash
pre-commit install
```

The hooks run:

- Ruff linting, with automatic fixes where possible
- Ruff formatting
- API documentation generation in `docs/api.md`
- Notebook documentation generation
- NumPy-style docstring checks with `pydoclint`
- A Zensical documentation build
- The full pytest suite

To run the same checks without making a commit:

```bash
pre-commit run --all-files
```

When the public API changes, update its exports and docstrings; the hooks generate `docs/api.md` from them.

## 2. Code

The `main` branch holds released code and accepts changes only through pull requests. Start each change from the main repository's `dev` branch, then open a pull request from your fork back to `dev`. Changes can accumulate there until the maintainers are ready to release a new version of the package.

```text
issue -> fork -> branch from upstream/dev -> develop and test -> push to fork -> PR to dev -> automatic checks and review -> merge to dev
```

Use branch names such as `feature/<issue>`, `fix/<issue>`, and `docs/<topic>`. Use [Conventional Commits](https://www.conventionalcommits.org/) for commit subjects; `fix:`, `feat:`, `deps:`, `docs:`, and `chore:` are the usual types.

Start a branch from the latest `dev`:

```bash
git fetch upstream
git switch -c fix/123-short-description upstream/dev
```

When the change is ready, add or update its tests and run:

```bash
pre-commit run --all-files
```

After committing the changes, push the branch to your fork:

```bash
git push -u origin fix/123-short-description
```

Then open a pull request to the `dev` branch of the main repository.

## 3. Submit a pull request

Opening a pull request to `dev` starts two workflows:

- **Package quality** builds the package and checks the code, documentation, and tests; it must pass before the pull request can merge.
- **Ecosystem** checks compatibility with the OPERA ecosystem packages; this check is optional.

## For maintainers

### 4. Release a version

When the changes in `dev` are ready to ship, a maintainer opens a pull request from `dev` to `main`. This starts the package-quality and ecosystem workflows again.

After the pull request merges, Release Please reads the new commits, updates the version and `CHANGELOG.md`, and opens a release pull request. Releases normally increment the patch version; to choose another version, add `Release-As: x.y.z` to the pull request message.

```text
Maintainer PR -> package-quality + ecosystem + review -> merge -> Release Please opens a release PR -> package-quality -> auto-merge -> GitHub Release
```

Publishing the GitHub Release starts the PyPI and documentation workflows. After PyPI publication succeeds, another workflow updates the `bvar` pin in `opera-eco`.

### The whole workflow

```text
------ Installation
-> fork and clone the repository
-> add the main repository as upstream
-> pip install -e ".[dev]"
-> pre-commit install

------ Development
-> branch from upstream/dev
-> develop and test
-> push the branch to your fork
-> open a PR from your fork to dev
-> package-quality and ecosystem run
-> review and merge to dev

------ Release
-> a maintainer opens a PR from dev to main
-> package-quality, ecosystem, and review
-> merge to main
-> Release Please opens and auto-merges a release PR
-> GitHub creates the version tag and release
-> documentation deploys to GitHub Pages
-> publish-pypi publishes the package
-> update-ecosystem updates the bvar pin in opera-eco
```

### Workflow summaries

- **Package quality** checks pull requests to `main` or `dev` by building the package, checking the code and documentation, and running the tests.
- **Ecosystem** runs the OPERA ecosystem contract and pipeline tests for pull requests to `main` or `dev`, except Release Please pull requests.
- **Release Please** runs after changes reach `main`, then opens or updates the release pull request and enables auto-merge when appropriate.
- **Publish to PyPI** builds and publishes the package after a release, then starts the ecosystem pin update.
- **Deploy documentation** builds and deploys the documentation to GitHub Pages after a release.
- **Update opera-eco pin** updates the pinned package version and generated APIs, then opens or updates an auto-merged pull request in `opera-eco`.
