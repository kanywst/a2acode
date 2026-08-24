# syntax=docker/dockerfile:1

# The `acp` backend launches its agent adapter with `npx`, so the runtime needs
# Node next to Python. Copied out of the official image rather than piped from a
# setup script; both stages are the same Debian release, so the binary matches.
#
# Both bases are pinned by digest as well as tag. The tag is republished
# whenever Debian patches the layers underneath it, so a digest is the only
# form of it Dependabot can see move — and the only one that builds the same
# image twice.
FROM node:24-trixie-slim@sha256:0711b541c1c33a8a530ac4f0d391baa9a15b3d804695b1b24a47daa5fb60e74d AS node

FROM python:3.13-slim-trixie@sha256:ffb752e139c0a19692a43af8d8523b274222dd68eebad5d583b45c2201c6e30a

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# The wheel `uv build` produced, so the image ships the artifact the release
# publishes rather than a second resolution of the same code. Every extra is
# installed: which backend and task store it runs is a flag, not a rebuild.
COPY dist/*.whl /tmp/dist/
RUN set -eu; \
    wheel="$(find /tmp/dist -name '*.whl')"; \
    test "$(printf '%s\n' "$wheel" | wc -l)" -eq 1; \
    pip install --no-cache-dir "${wheel}[claude,telemetry,persistence]"; \
    rm -rf /tmp/dist

# A coding agent runs whatever the caller approves, so it does not also get to be
# root. npx caches its downloads under HOME, which has to belong to that user.
RUN useradd --create-home --uid 1000 a2acode
ENV HOME=/home/a2acode
USER a2acode

# --cwd defaults to the working directory, so mounting a project over /workspace
# is all it takes to point the agent at one.
WORKDIR /workspace
EXPOSE 9100

# Entering on the bare command keeps `call` and `card` reachable for debugging
# from inside the same network. --host is in the default command because the
# CLI's own default is loopback, which no one outside the container can reach.
ENTRYPOINT ["a2acode"]
CMD ["serve", "--host", "0.0.0.0"]
