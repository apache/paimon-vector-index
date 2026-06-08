#!/usr/bin/env python3

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Validate .asf.yaml structural correctness.

Checks YAML syntax, root-level structure, and type constraints for
known fields. Does NOT reject unknown keys — the ASF infra schema
evolves independently and unrecognized keys are silently ignored by
the ASF gitbox service.

Reference: https://github.com/apache/infrastructure-asfyaml
"""

import sys

try:
    import yaml
except ImportError:
    sys.exit(
        "PyYAML is required: pip install pyyaml\n"
        "Or use: python3 -c \"import pip; pip.main(['install', 'pyyaml'])\""
    )

ASF_YAML_PATH = ".asf.yaml"


def validate():
    errors = []

    try:
        with open(ASF_YAML_PATH, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"SKIP: {ASF_YAML_PATH} not found")
        return 0
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML syntax in {ASF_YAML_PATH}: {e}")
        return 1

    if not isinstance(data, dict):
        print(f"ERROR: {ASF_YAML_PATH} root must be a mapping")
        return 1

    # Validate types of known sections
    for key in ("github", "notifications", "staging", "publish"):
        if key in data and not isinstance(data[key], dict):
            errors.append(f"'{key}' must be a mapping, got {type(data[key]).__name__}")

    github = data.get("github")
    if isinstance(github, dict):
        # features values must be booleans
        features = github.get("features")
        if isinstance(features, dict):
            for key, val in features.items():
                if not isinstance(val, bool):
                    errors.append(
                        f"'github.features.{key}' must be a boolean, "
                        f"got {type(val).__name__}"
                    )

        # enabled_merge_buttons values must be booleans or strings
        merge_buttons = github.get("enabled_merge_buttons")
        if isinstance(merge_buttons, dict):
            for key, val in merge_buttons.items():
                if not isinstance(val, (bool, str)):
                    errors.append(
                        f"'github.enabled_merge_buttons.{key}' must be bool or string, "
                        f"got {type(val).__name__}"
                    )

        # labels must be a list of strings
        labels = github.get("labels")
        if labels is not None and not isinstance(labels, list):
            errors.append(f"'github.labels' must be a list, got {type(labels).__name__}")

    if errors:
        print(f"ERROR: {ASF_YAML_PATH} validation failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"OK: {ASF_YAML_PATH} is valid")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
