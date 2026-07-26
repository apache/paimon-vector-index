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

import java.util.HashMap;
import java.util.Map;

/** Smoke test used with the packaged multi-platform JAR as the first classpath entry. */
public class VectorIndexNativeLoaderSmokeTest {

    public static void main(String[] args) {
        configureExternalLibrary(args);

        Map<String, String> options = new HashMap<>();
        options.put("index.type", "ivf_flat");
        options.put("dimension", "2");
        options.put("nlist", "1");
        options.put("metric", "l2");

        try (VectorIndexTrainer ignored = VectorIndexTrainer.create(options)) {
            // Creating the native trainer proves that the JNI methods are linked.
        }
    }

    static void configureExternalLibrary(String[] args) {
        if (args.length > 1) {
            throw new IllegalArgumentException(
                    "expected zero arguments or one native library path");
        }
        if (args.length == 1) {
            System.setProperty(NativeLibraryLoader.NATIVE_PATH_PROPERTY, args[0]);
        }
    }
}
