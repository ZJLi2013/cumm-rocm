#pragma once
#include <cstdint>
#include <limits>
#include <type_traits>

// Self-contained GPU hash table for spconv indice pairs.
// Extracted and simplified from tensorview's LinearHashTableSplit.
// - Open-addressing with linear probing
// - Split layout: keys and values in separate arrays (cache-friendly)
// - Murmur3 hash function

namespace spconv_hip {

// Atomic CAS wrapper: HIP atomicCAS only supports unsigned int / unsigned long long.
template <typename K>
__device__ __forceinline__ K atomic_cas(K* addr, K compare, K val);

template <>
__device__ __forceinline__ int32_t atomic_cas<int32_t>(int32_t* addr, int32_t compare, int32_t val) {
    return (int32_t)atomicCAS((unsigned int*)addr, (unsigned int)compare, (unsigned int)val);
}

template <>
__device__ __forceinline__ int64_t atomic_cas<int64_t>(int64_t* addr, int64_t compare, int64_t val) {
    return (int64_t)atomicCAS((unsigned long long*)addr, (unsigned long long)compare, (unsigned long long)val);
}

template <typename K>
struct Murmur3Hash {
    __host__ __device__ __forceinline__
    K operator()(K key) const {
        key ^= key >> 16;
        key *= 0x85ebca6b;
        key ^= key >> 13;
        key *= 0xc2b2ae35;
        key ^= key >> 16;
        return key;
    }
};

template <>
struct Murmur3Hash<int64_t> {
    __host__ __device__ __forceinline__
    int64_t operator()(int64_t key) const {
        key ^= key >> 33;
        key *= 0xff51afd7ed558ccdLL;
        key ^= key >> 33;
        key *= 0xc4ceb9fe1a85ec53LL;
        key ^= key >> 33;
        return key;
    }
};

// Open-addressing hash table with linear probing.
// K = key type (int32_t or int64_t), V = value type (int32_t).
// Keys and values stored in separate arrays for better cache behavior
// during key-only operations (insert, lookup).
template <typename K, typename V>
struct LinearHashTable {
    K* keys;
    V* values;
    int capacity;
    static constexpr K empty_key = std::numeric_limits<K>::max();

    __host__ __device__
    LinearHashTable(K* keys_, V* values_, int capacity_)
        : keys(keys_), values(values_), capacity(capacity_) {}

    __device__ __forceinline__
    void insert(K key, V value) {
        Murmur3Hash<K> hasher;
        using UK = typename std::make_unsigned<K>::type;
        int slot = (int)((UK)hasher(key) % (UK)capacity);
        while (true) {
            K prev = atomic_cas<K>(&keys[slot], empty_key, key);
            if (prev == empty_key || prev == key) {
                values[slot] = value;
                return;
            }
            slot = (slot + 1) % capacity;
        }
    }

    __device__ __forceinline__
    int lookup_offset(K key) const {
        Murmur3Hash<K> hasher;
        using UK = typename std::make_unsigned<K>::type;
        int slot = (int)((UK)hasher(key) % (UK)capacity);
        while (true) {
            K cur = keys[slot];
            if (cur == key) return slot;
            if (cur == empty_key) return -1;
            slot = (slot + 1) % capacity;
        }
    }
};

// Fill key array with empty sentinel.
template <typename K>
__global__ void clear_hash_table_kernel(K* keys, int capacity) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < capacity;
         i += gridDim.x * blockDim.x) {
        keys[i] = std::numeric_limits<K>::max();
    }
}

}  // namespace spconv_hip
