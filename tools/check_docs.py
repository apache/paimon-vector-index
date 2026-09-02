#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.

"""Validate local links and anchors in the generated documentation site."""

import argparse
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


SITE_PREFIX = "/docs/vector-index/"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.duplicate_ids = set()
        self.references = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            identifier = attributes["id"]
            if identifier in self.ids:
                self.duplicate_ids.add(identifier)
            self.ids.add(identifier)
        for name in ("href", "src"):
            if name in attributes:
                self.references.append(attributes[name])


def parse_page(path: Path) -> PageParser:
    parser = PageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def resolve_reference(site: Path, source: Path, reference: str):
    parts = urlsplit(reference)
    if parts.scheme or parts.netloc or reference.startswith(("mailto:", "data:")):
        return None

    path = unquote(parts.path)
    if path.startswith(SITE_PREFIX):
        target = site / path.removeprefix(SITE_PREFIX)
    elif path.startswith("/"):
        return None
    elif path:
        target = source.parent / path
    else:
        target = source

    if path.endswith("/") or target.is_dir():
        target /= "index.html"
    elif not target.exists() and not target.suffix:
        directory_index = target / "index.html"
        html_file = target.with_suffix(".html")
        if directory_index.exists():
            target = directory_index
        elif html_file.exists():
            target = html_file
    return target.resolve(), unquote(parts.fragment)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("site", type=Path)
    args = parser.parse_args()

    site = args.site.resolve()
    pages = {
        path.resolve(): parse_page(path)
        for path in site.rglob("*.html")
        if path.is_file()
    }
    failures = []

    for source, page in pages.items():
        for identifier in sorted(page.duplicate_ids):
            failures.append(f"{source.relative_to(site)}: duplicate id: {identifier}")
        for reference in page.references:
            resolved = resolve_reference(site, source, reference)
            if resolved is None:
                continue
            target, fragment = resolved
            try:
                target.relative_to(site)
            except ValueError:
                failures.append(f"{source.relative_to(site)}: link escapes site: {reference}")
                continue
            if not target.exists():
                failures.append(f"{source.relative_to(site)}: missing target: {reference}")
                continue
            if fragment and target.suffix == ".html":
                target_page = pages.get(target)
                if target_page is None or fragment not in target_page.ids:
                    failures.append(f"{source.relative_to(site)}: missing anchor: {reference}")

    if failures:
        print("Documentation link check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Checked {len(pages)} HTML pages: all local links and anchors resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
