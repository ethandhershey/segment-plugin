# Project Name

Brief description of what this project does.

## Installation

```bash
pip install segment-plugin
```

## Usage

### Basic Example

```python
# Add your basic usage example here
```

### Advanced Usage

```python
# Add more complex examples here
```

## Features

- Feature 1
- Feature 2
- Feature 3

## Requirements

- Python 3.8+
- Additional requirements...

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/ethandhershey/segment-plugin.git
cd your-repo

# Install development dependencies
pip install -e .
```

### Building

```bash
# Build the extension
maturin develop

# Or build wheels
maturin build --release
```

### Testing

```bash
# Run tests
pytest
```

## Release Process

This project uses automated versioning and publishing. Here's how to create a new release:

### Prerequisites

1. Install `bump-my-version`:
   ```bash
   pip install bump-my-version
   ```

2. Make sure all your changes are committed and pushed to `main`

### Creating a Release

1. **Finalize your changes** - Make sure your final patch is committed and pushed:
   ```bash
   git add .
   git commit -m "Final changes for release"
   git push origin main
   ```

2. **Bump the version** - This automatically updates `Cargo.toml`, creates a commit, and tags the release:
   ```bash
   # For bug fixes
   bump-my-version bump patch    # 0.1.0 → 0.1.1
   
   # For new features
   bump-my-version bump minor    # 0.1.0 → 0.2.0
   
   # For breaking changes
   bump-my-version bump major    # 0.1.0 → 1.0.0
   ```

3. **Push the tag** - This triggers automatic publishing to PyPI:
   ```bash
   git push origin main --tags
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for your changes
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a history of changes to this project.

## Support

- Create an issue on [GitHub Issues](../../issues)
- For questions, use [GitHub Discussions](../../discussions)

## Acknowledgments

- Credit contributors
- Credit inspiration/libraries used