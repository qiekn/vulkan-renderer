#!/bin/bash
# Fetch tutorial chapter sources (.adoc) from the upstream Vulkan-Tutorial repo.
#
# The rendered site at docs.vulkan.org sits behind a Cloudflare challenge that
# blocks scraping, so we pull the AsciiDoc sources directly from GitHub.
#
# Usage:
#   ./fetch.sh <section>            # fetch all .adoc files under a section dir
#   ./fetch.sh <section>/<file>     # fetch a single chapter file
#
# Examples:
#   ./fetch.sh Engine_Architecture
#   ./fetch.sh Camera_Transformations
#   ./fetch.sh Engine_Architecture/03_component_systems.adoc

set -euo pipefail

REPO="KhronosGroup/Vulkan-Tutorial"
BRANCH="main"
UPSTREAM_BASE="en/Building_a_Simple_Engine"
RAW_BASE="https://raw.githubusercontent.com/${REPO}/${BRANCH}/${UPSTREAM_BASE}"
API_BASE="https://api.github.com/repos/${REPO}/contents/${UPSTREAM_BASE}"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <section>[/<file.adoc>]" >&2
  exit 1
fi

target="$1"

# Single file: "Engine_Architecture/03_component_systems.adoc"
if [[ "${target}" == *.adoc ]]; then
  mkdir -p "$(dirname "${target}")"
  curl -fsSL "${RAW_BASE}/${target}" -o "${target}"
  echo "Fetched ${target}"
  exit 0
fi

# Whole section: list via GitHub API, then fetch each .adoc
mkdir -p "${target}"
curl -fsSL "${API_BASE}/${target}" \
  | tr ',' '\n' \
  | grep -E '"name":.*\.adoc"' \
  | sed -E 's/.*"name": *"([^"]+)".*/\1/' \
  | while read -r name; do
      curl -fsSL "${RAW_BASE}/${target}/${name}" -o "${target}/${name}"
      echo "Fetched ${target}/${name}"
    done

# vim: ft=sh ts=2 sw=2 et
