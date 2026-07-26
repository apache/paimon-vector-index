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

/**
 * Builds a trained vector-index state from one or more training batches.
 *
 * <p>Staging batches avoids requiring one large Java {@code float[]} and its array length limit.
 * Native deterministic reservoir sampling retains at most the automatically selected training
 * sample, independent of batch boundaries.
 */
public final class VectorIndexTrainer implements AutoCloseable {

    private final Object nativeHandleLock = new Object();
    private long nativePtr;
    private Thread nativeHandleOwner;

    private VectorIndexTrainer(long nativePtr) {
        this.nativePtr = nativePtr;
    }

    public static VectorIndexTrainer create(Map<String, String> options) {
        if (options == null) {
            throw new NullPointerException("options");
        }
        String[] keys = new String[options.size()];
        String[] values = new String[options.size()];
        int index = 0;
        for (Map.Entry<String, String> entry : options.entrySet()) {
            keys[index] = entry.getKey();
            values[index] = entry.getValue();
            index++;
        }
        return new VectorIndexTrainer(VectorIndexNative.createTrainer(keys, values));
    }

    public static VectorIndexTraining train(
            Map<String, String> options, float[] data, int vectorCount) {
        if (options == null) {
            throw new NullPointerException("options");
        }
        if (data == null) {
            throw new NullPointerException("data");
        }
        if (vectorCount <= 0 || data.length % vectorCount != 0) {
            throw new IllegalArgumentException(
                    "data length must be divisible by a positive vectorCount");
        }
        Map<String, String> resolved = new HashMap<>(options);
        if (!resolved.containsKey("dimension") || "auto".equals(resolved.get("dimension"))) {
            resolved.put("dimension", Integer.toString(data.length / vectorCount));
        }
        String indexType = resolved.getOrDefault("index.type", "");
        if (indexType.startsWith("ivf_")
                && (!resolved.containsKey("nlist") || "auto".equals(resolved.get("nlist")))
                && !resolved.containsKey("expected-vector-count")) {
            resolved.put("expected-vector-count", Integer.toString(vectorCount));
        }
        try (VectorIndexTrainer trainer = create(resolved)) {
            return trainer.addTrainingVectors(data, vectorCount).finishTraining();
        }
    }

    static VectorIndexTrainer fromNativePointerForTesting(long nativePtr) {
        return new VectorIndexTrainer(nativePtr);
    }

    public int dimension() {
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.trainerDimension(requireOpen());
            } finally {
                exitNativeHandle();
            }
        }
    }

    public VectorIndexTrainer addTrainingVectors(float[] data, int vectorCount) {
        if (data == null) {
            throw new NullPointerException("data");
        }
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                VectorIndexNative.trainerAddTrainingVectors(requireOpen(), data, vectorCount);
                return this;
            } finally {
                exitNativeHandle();
            }
        }
    }

    /**
     * Finishes training and returns a trained state for creating a {@link VectorIndexWriter}.
     *
     * <p>This method consumes this trainer on both success and failure. After it returns or throws,
     * this trainer is closed and cannot accept more training batches; create a new trainer to retry.
     */
    public VectorIndexTraining finishTraining() {
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                long ptr = requireOpen();
                nativePtr = 0L;
                return new VectorIndexTraining(VectorIndexNative.trainerFinishTraining(ptr));
            } finally {
                exitNativeHandle();
            }
        }
    }

    @Override
    public void close() {
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                long ptr = nativePtr;
                nativePtr = 0L;
                if (ptr != 0L) {
                    VectorIndexNative.freeTrainer(ptr);
                }
            } finally {
                exitNativeHandle();
            }
        }
    }

    private long requireOpen() {
        if (nativePtr == 0L) {
            throw new IllegalStateException("VectorIndexTrainer is closed");
        }
        return nativePtr;
    }

    private void enterNativeHandle() {
        Thread current = Thread.currentThread();
        if (nativeHandleOwner == current) {
            throw new IllegalStateException("VectorIndexTrainer native handle is already in use");
        }
        nativeHandleOwner = current;
    }

    private void exitNativeHandle() {
        nativeHandleOwner = null;
    }
}
