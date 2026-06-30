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

public final class VectorIndexTraining implements AutoCloseable {

    private final Object nativeHandleLock = new Object();
    private long nativePtr;
    private Thread nativeHandleOwner;

    VectorIndexTraining(long nativePtr) {
        this.nativePtr = nativePtr;
    }

    static VectorIndexTraining fromNativePointerForTesting(long nativePtr) {
        return new VectorIndexTraining(nativePtr);
    }

    long takeNativePointer() {
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                long ptr = requireOpen();
                nativePtr = 0L;
                return ptr;
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
                    VectorIndexNative.freeTraining(ptr);
                }
            } finally {
                exitNativeHandle();
            }
        }
    }

    private long requireOpen() {
        if (nativePtr == 0L) {
            throw new IllegalStateException("VectorIndexTraining is closed");
        }
        return nativePtr;
    }

    private void enterNativeHandle() {
        Thread current = Thread.currentThread();
        if (nativeHandleOwner == current) {
            throw new IllegalStateException("VectorIndexTraining native handle is already in use");
        }
        nativeHandleOwner = current;
    }

    private void exitNativeHandle() {
        nativeHandleOwner = null;
    }
}
