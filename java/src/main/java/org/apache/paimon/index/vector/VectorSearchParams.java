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

public final class VectorSearchParams {

    private final int topK;
    private final int nprobe;
    private final int efSearch;
    private final int queryBits;

    public VectorSearchParams(int topK, int nprobe) {
        this(topK, nprobe, 0, 0);
    }

    public VectorSearchParams(int topK, int nprobe, int efSearch, int queryBits) {
        this.topK = topK;
        this.nprobe = nprobe;
        this.efSearch = efSearch;
        this.queryBits = queryBits;
    }

    public int topK() {
        return topK;
    }

    public int nprobe() {
        return nprobe;
    }

    public int efSearch() {
        return efSearch;
    }

    public int queryBits() {
        return queryBits;
    }

    public VectorSearchParams withEfSearch(int efSearch) {
        return new VectorSearchParams(topK, nprobe, efSearch, queryBits);
    }

    public VectorSearchParams withQueryBits(int queryBits) {
        return new VectorSearchParams(topK, nprobe, efSearch, queryBits);
    }
}
