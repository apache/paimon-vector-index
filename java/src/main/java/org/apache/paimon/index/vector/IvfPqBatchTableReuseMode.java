// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package org.apache.paimon.index.vector;

import java.util.Locale;

public enum IvfPqBatchTableReuseMode {
    OFF(0),
    ON(1),
    AUTO(2);

    private final int code;

    IvfPqBatchTableReuseMode(int code) {
        this.code = code;
    }

    int code() {
        return code;
    }

    static IvfPqBatchTableReuseMode fromCode(int code) {
        switch (code) {
            case 0:
                return OFF;
            case 1:
                return ON;
            case 2:
                return AUTO;
            default:
                throw new IllegalArgumentException(
                        "Unknown IVF-PQ batch table reuse mode code: " + code);
        }
    }

    public static IvfPqBatchTableReuseMode fromString(String value) {
        if (value == null) {
            throw new IllegalArgumentException("IVF-PQ batch table reuse mode is null");
        }
        switch (value.toLowerCase(Locale.ROOT)) {
            case "off":
                return OFF;
            case "on":
                return ON;
            case "auto":
                return AUTO;
            default:
                throw new IllegalArgumentException(
                        "Invalid IVF-PQ batch table reuse mode '"
                                + value
                                + "'. Expected off, on, or auto.");
        }
    }
}
