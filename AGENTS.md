# Contributor guide

## Development environment tips
- Use `conda activate synthetic_pages` to activate the conda environment. This contains all of the necessary packages
- Use `cat` to display whole files
- Use `find` and `grep` to search files
- Refer to `README.md` for information about this repository

## Testing instructions
- Use `python -m unittest` to run test files
- Test files are contained in the `tests/` directory
- Separate classes should have separate test files
- Use the `unittest` module to create tests

## Pull request guide
- Include unit tests demonstrating that intended functionality is achieved.
- Bug fixes should include regression tests that fail before the provided fix and pass afterwards
