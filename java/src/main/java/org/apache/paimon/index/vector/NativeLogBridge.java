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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Receives diagnostic records from the Rust core (installed by JNI_OnLoad in
 * jni/src/log_bridge.rs) and forwards them to SLF4J. Package-private is
 * sufficient: JNI method lookup does not enforce Java access control.
 */
final class NativeLogBridge {

    private static final Logger LOG = LoggerFactory.getLogger(NativeLogBridge.class);

    private NativeLogBridge() {}

    // Called from native code; the signature must remain (ILjava/lang/String;)V.
    // Must never throw: a pending exception here would surface on arbitrary
    // native threads.
    static void log(int level, String message) {
        try {
            switch (level) {
                case 1:
                    LOG.error(message);
                    break;
                case 2:
                    LOG.warn(message);
                    break;
                case 4:
                    LOG.debug(message);
                    break;
                case 3:
                default:
                    LOG.info(message);
                    break;
            }
        } catch (Throwable ignored) {
            try {
                System.err.println(message);
            } catch (Throwable ignoredFallback) {
                // Keep the JNI callback non-throwing even if stderr itself fails.
            }
        }
    }
}
