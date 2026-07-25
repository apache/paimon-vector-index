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

public final class VectorIndexMetadata {

    private final String indexType;
    private final int dimension;
    private final int nlist;
    private final String metric;
    private final long totalVectors;
    private final int pqM;
    private final int pqBits;
    private final int rqBits;
    private final int diskAnnMaxDegree;
    private final int diskAnnBuildSearchListSize;
    private final float diskAnnAlpha;

    public VectorIndexMetadata(
            String indexType,
            int dimension,
            int nlist,
            String metric,
            long totalVectors,
            int pqM,
            int pqBits,
            int rqBits,
            int diskAnnMaxDegree,
            int diskAnnBuildSearchListSize,
            float diskAnnAlpha) {
        if (indexType == null) {
            throw new NullPointerException("indexType");
        }
        if (metric == null) {
            throw new NullPointerException("metric");
        }
        this.indexType = indexType;
        this.dimension = dimension;
        this.nlist = nlist;
        this.metric = metric;
        this.totalVectors = totalVectors;
        this.pqM = pqM;
        this.pqBits = pqBits;
        this.rqBits = rqBits;
        this.diskAnnMaxDegree = diskAnnMaxDegree;
        this.diskAnnBuildSearchListSize = diskAnnBuildSearchListSize;
        this.diskAnnAlpha = diskAnnAlpha;
    }

    public String indexType() {
        return indexType;
    }

    public int dimension() {
        return dimension;
    }

    public int nlist() {
        return nlist;
    }

    public String metric() {
        return metric;
    }

    public long totalVectors() {
        return totalVectors;
    }

    public int pqM() {
        return pqM;
    }

    public int pqBits() {
        return pqBits;
    }

    public int rqBits() {
        return rqBits;
    }

    public int diskAnnMaxDegree() {
        return diskAnnMaxDegree;
    }

    public int diskAnnBuildSearchListSize() {
        return diskAnnBuildSearchListSize;
    }

    public float diskAnnAlpha() {
        return diskAnnAlpha;
    }

}
