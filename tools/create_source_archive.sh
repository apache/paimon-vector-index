#!/usr/bin/env bash

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

set -o errexit
set -o nounset
set -o pipefail

if [[ $# -ne 2 || -z "$1" || -z "$2" ]]; then
  echo "Usage: create_source_archive.sh RELEASE_VERSION OUTPUT_FILE" >&2
  exit 1
fi

RELEASE_VERSION=$1
OUTPUT_FILE=$2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# Archive the commit, rather than its tree object, so Git uses the stable
# commit timestamp instead of the current time for tar headers.
git -C "$REPO_DIR" archive \
  --format=tar \
  --prefix="paimon-vector-index-${RELEASE_VERSION}/" \
  HEAD . \
  ':(exclude).gitignore' ':(exclude).gitattributes' \
  ':(exclude).asf.yaml' ':(exclude).github' \
  ':(exclude)deploysettings.xml' ':(exclude)target' \
  ':(exclude).idea' ':(exclude)*.iml' ':(exclude).DS_Store' \
  | gzip -n > "$OUTPUT_FILE"
