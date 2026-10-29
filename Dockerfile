# CPU-only container image for kazner.
#
#   docker build --target test -t kazner:test .     # offline test suite (default command)
#   docker build -t kazner .                        # runtime image, entry point `kazner`
#
# Stages: base (Python + CPU torch + runtime dependencies) -> test (dev extras, sources, tests)
# and wheel (builds the package) -> runtime (base + the wheel only, non-root user).
# Dependencies are installed before the sources are copied, so a code change reuses the
# cached torch and dependency layers.

ARG PYTHON_IMAGE=python:3.11-slim-bookworm

FROM ${PYTHON_IMAGE} AS base
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# CPU wheels only: the default PyPI torch wheel for Linux bundles CUDA (several GB).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install "torch>=2.5,<3" --index-url "${TORCH_INDEX}"
WORKDIR /src
# Install the pinned runtime dependencies from pyproject.toml through a stub package,
# then remove the stub: this layer changes only when the dependency list changes.
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src/kazner && touch src/kazner/__init__.py \
    && pip install . \
    && pip uninstall -y kazner \
    && rm -rf src build

FROM base AS test
# git: the run-ID code reads tree hashes, and its tests create temporary repositories.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p src/kazner && touch src/kazner/__init__.py \
    && pip install ".[dev]" \
    && pip uninstall -y kazner \
    && rm -rf src build
COPY . .
RUN pip install --no-deps -e .
CMD ["python", "-m", "pytest", "-m", "not network and not smoke", "-q"]

FROM base AS wheel
COPY src ./src
RUN pip wheel --no-deps --wheel-dir /dist .

FROM base AS runtime
LABEL org.opencontainers.image.title="kazner" \
      org.opencontainers.image.description="Kazakh NER toolkit (CPU)" \
      org.opencontainers.image.source="https://github.com/DiasZhaga/kazner" \
      org.opencontainers.image.licenses="MIT"
COPY --from=wheel /dist/*.whl /tmp/
RUN pip install --no-deps /tmp/*.whl && rm /tmp/*.whl
# A synthetic prediction file, so the image can be checked without any data.
COPY examples/contract_example.csv /opt/kazner/examples/
RUN useradd --create-home --uid 10001 kazner
USER kazner
WORKDIR /work
ENTRYPOINT ["kazner"]
CMD ["--help"]
