# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

cmake_minimum_required(VERSION 3.16)
project(paimon-vindex-cpp-test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PAIMON_VINDEX_FFI_LIB "" CACHE FILEPATH "Path to libpaimon_vindex_ffi shared library")
if(NOT PAIMON_VINDEX_FFI_LIB)
    if(APPLE)
        set(PAIMON_VINDEX_FFI_LIB "${CMAKE_SOURCE_DIR}/../target/release/libpaimon_vindex_ffi.dylib")
    elseif(WIN32)
        set(PAIMON_VINDEX_FFI_LIB "${CMAKE_SOURCE_DIR}/../target/release/paimon_vindex_ffi.dll")
    else()
        set(PAIMON_VINDEX_FFI_LIB "${CMAKE_SOURCE_DIR}/../target/release/libpaimon_vindex_ffi.so")
    endif()
endif()

add_executable(test_vindex_cpp test_vindex.cpp)
target_include_directories(test_vindex_cpp PRIVATE ${CMAKE_SOURCE_DIR}/../include)
target_link_libraries(test_vindex_cpp ${PAIMON_VINDEX_FFI_LIB})
