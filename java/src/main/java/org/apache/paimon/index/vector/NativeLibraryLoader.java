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

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

/** Loads the JNI library from an explicit path or the platform resource bundled in the JAR. */
final class NativeLibraryLoader {

    static final String NATIVE_PATH_PROPERTY = "paimon.vindex.native.path";

    private static boolean loaded;

    private NativeLibraryLoader() {}

    static synchronized void load() {
        if (loaded) {
            return;
        }

        String explicitPath = System.getProperty(NATIVE_PATH_PROPERTY);
        if (explicitPath != null && !explicitPath.trim().isEmpty()) {
            System.load(Paths.get(explicitPath).toAbsolutePath().normalize().toString());
            loaded = true;
            return;
        }

        String resourcePath =
                resourcePath(
                        System.getProperty("os.name"), System.getProperty("os.arch"));
        String fileName = resourcePath.substring(resourcePath.lastIndexOf('/') + 1);
        int extension = fileName.lastIndexOf('.');
        String suffix = extension >= 0 ? fileName.substring(extension) : null;

        try (InputStream input = NativeLibraryLoader.class.getResourceAsStream(resourcePath)) {
            if (input == null) {
                throw loadError(
                        "Bundled JNI library is missing: "
                                + resourcePath
                                + ". Set -D"
                                + NATIVE_PATH_PROPERTY
                                + "=/absolute/path/to/the/library to use an external build.",
                        null);
            }

            Path extracted = Files.createTempFile("paimon-vindex-jni-", suffix);
            extracted.toFile().deleteOnExit();
            Files.copy(input, extracted, StandardCopyOption.REPLACE_EXISTING);
            System.load(extracted.toAbsolutePath().toString());
            loaded = true;
        } catch (IOException e) {
            throw loadError("Failed to extract bundled JNI library " + resourcePath, e);
        }
    }

    static String resourcePath(String osName, String osArch) {
        String os = normalizeOs(osName);
        String arch = normalizeArch(osArch);
        String fileName;

        if ("linux".equals(os)) {
            fileName = "libpaimon_vindex_jni.so";
        } else if ("macos".equals(os)) {
            fileName = "libpaimon_vindex_jni.dylib";
        } else if ("windows".equals(os)) {
            fileName = "paimon_vindex_jni.dll";
        } else {
            throw loadError("Unsupported operating system: " + osName, null);
        }

        if (("macos".equals(os) && !"aarch64".equals(arch))
                || ("windows".equals(os) && !"x86_64".equals(arch))) {
            throw loadError(
                    "No bundled JNI library for "
                            + osName
                            + " / "
                            + osArch
                            + ". Set -D"
                            + NATIVE_PATH_PROPERTY
                            + "=/absolute/path/to/the/library to use an external build.",
                    null);
        }

        return "/native/" + os + "/" + arch + "/" + fileName;
    }

    private static String normalizeOs(String osName) {
        String value = osName == null ? "" : osName.toLowerCase(Locale.ROOT);
        if (value.contains("linux")) {
            return "linux";
        }
        if (value.contains("mac") || value.contains("darwin")) {
            return "macos";
        }
        if (value.contains("windows")) {
            return "windows";
        }
        return value;
    }

    private static String normalizeArch(String osArch) {
        String value = osArch == null ? "" : osArch.toLowerCase(Locale.ROOT);
        if ("amd64".equals(value) || "x86_64".equals(value) || "x64".equals(value)) {
            return "x86_64";
        }
        if ("aarch64".equals(value) || "arm64".equals(value)) {
            return "aarch64";
        }
        throw loadError("Unsupported CPU architecture: " + osArch, null);
    }

    private static UnsatisfiedLinkError loadError(String message, Throwable cause) {
        UnsatisfiedLinkError error = new UnsatisfiedLinkError(message);
        if (cause != null) {
            error.initCause(cause);
        }
        return error;
    }
}
