// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.



package org.apache.paimon.index.vector;

/** Immutable per-query diagnostics, independent of the reader's last Top-K statistics. */
public final class DiskAnnRangeSearchStats {
    private final long nodesExpanded;
    private final long pqEvaluations;
    private final long vectorEvaluations;
    private final long readCalls;
    private final long readBytes;
    private final boolean secondPhase;
    private final boolean exhaustive;

    DiskAnnRangeSearchStats(long[] values, int offset) {
        nodesExpanded = values[offset];
        pqEvaluations = values[offset + 1];
        vectorEvaluations = values[offset + 2];
        readCalls = values[offset + 3];
        readBytes = values[offset + 4];
        secondPhase = values[offset + 5] != 0;
        exhaustive = values[offset + 6] != 0;
    }

    public long nodesExpanded() { return nodesExpanded; }
    public long pqEvaluations() { return pqEvaluations; }
    public long vectorEvaluations() { return vectorEvaluations; }
    public long readCalls() { return readCalls; }
    public long readBytes() { return readBytes; }
    public boolean secondPhase() { return secondPhase; }
    public boolean exhaustive() { return exhaustive; }
}
