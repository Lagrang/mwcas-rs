[![Crate](https://img.shields.io/crates/v/mwcas.svg)](https://crates.io/crates/mwcas)
[![API](https://docs.rs/mwcas/badge.svg)](https://docs.rs/mwcas)

# Multi-word CAS.

Rust standard library provides atomic types in
`atomic` package. Atomic types provide lock-free way
to atomically update value of one pointer. Many concurrent data
structures usually requires atomic update for more than 1 pointer at
time. For example,
BzTree described in [`paper`](http://www.vldb.org/pvldb/vol11/p553-arulraj.pdf)([implementation](https://github.com/Lagrang/bztree-rs) in Rust).

This crate provides concurrency primitive called `MwCas` which can
atomically update several pointers at time. It is based on paper
[`Easy Lock-Free Indexing in Non-Volatile Memory`](http://justinlevandoski.org/papers/ICDE18_mwcas.pdf).
Current implementation doesn't provide features for non-volatile
memory(persistent memory) and only covers DRAM multi-word CAS.

## Platform support
Currently, `MwCas` supports only x86_64 platform because it exploits
platform specific hacks: `MwCas` use upper 3 bit of pointer's virtual
address to representing internal state. Today x86_64 CPUs use lower 48
bits of virtual address, other 16 bits are 0. Usage of upper 3 bits
described in paper.

## Usage
Multi-word CAS API represented by `MwCas` struct which can operate on 2
types of pointers:
- pointer to heap allocated data(`HeapPointer`)
- pointer to u64(`U64Pointer`)

`HeapPointer` can be used to execute multi-word CAS on any data type,
but with cost of heap allocation. `U64Pointer` do not allocate anything
on heap and has memory overhead same as `u64`.

`MwCas` is a container for chain of `compare_exchange` operations. When
caller adds all required CASes, it performs multi-word CAS by calling
`exec` method. `exec` method returns `bool` which indicate is MwCAS was
successful.

Example of `MwCAS` usage:
```
use mwcas::{MwCas, HeapPointer, U64Pointer};

let ptr = HeapPointer::new(String::new());
let val = U64Pointer::new(0);
let guard = crossbeam_epoch::pin();
let cur_ptr_val: &String = ptr.read(&guard);

let mut mwcas = MwCas::new();
mwcas.compare_exchange(&ptr, cur_ptr_val, String::from("new_string"));
mwcas.compare_exchange_u64(&val, 0, 1);

assert!(mwcas.exec(&guard));
assert_eq!(ptr.read(&guard), &String::from("new_string"));
assert_eq!(val.read(&guard), 1);
```

## Memory reclamation
Drop of values pointed by `HeapPointer` which were replaced by new one
during CAS, will be performed by [`crossbeam_epoch`](https://github.com/crossbeam-rs/crossbeam/tree/master/crossbeam-epoch) memory
reclamation.
