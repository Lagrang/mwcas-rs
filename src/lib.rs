//! Multi-word CAS.
//!
//! Rust standard library provides atomic types in [`atomic`](std::sync::atomic) package.
//! Atomic types provide lock-free way to atomically update value of one pointer. Many concurrent
//! data structures usually requires atomic update for more than 1 pointer at time. For example,
//! [`BzTree`](http://www.vldb.org/pvldb/vol11/p553-arulraj.pdf).
//!
//! This crate provides concurrency primitive called [`MwCas`] which can atomically update several
//! pointers at time. It is based on paper
//! [`Easy Lock-Free Indexing in Non-Volatile Memory`](http://justinlevandoski.org/papers/ICDE18_mwcas.pdf).
//! Current implementation doesn't provide features for non-volatile memory(persistent
//! memory) and only covers DRAM multi-word CAS.
//!
//! # Platform support
//! Currently, [`MwCas`] supports only x86_64 and ARMv8 platforms because it exploits platform specific hacks:
//! MwCas use upper 3 bit of pointer's virtual address to representing internal state. Today x86_64
//! and ARMv8 CPUs use lower 48 bits of virtual address, other 16 bits are 0. Usage of upper 3 bits
//! described in paper.
//!
//! # Usage
//! Multi-word CAS API represented by [`MwCas`] struct which can operate on 2 types of pointers:
//! - pointer to heap allocated data([`HeapPointer`])
//! - pointer to u64([`U64Pointer`])
//!
//! [`HeapPointer`] can be used to execute multi-word CAS on any data type, but with
//! cost of heap allocation. [`U64Pointer`] do not allocate anything on heap and has memory
//! overhead same as `u64`.
//!
//! [`MwCas`] is a container for chain of `compare_exchange` operations. When caller adds all
//! required CASes, it performs multi-word CAS by calling `exec` method. `exec` method
//! returns `bool` which indicate is MwCAS was successful.
//!
//! Example of `MwCAS` usage:
//! ```
//! use mwcas::{MwCas, HeapPointer, U64Pointer};
//!
//! let ptr = HeapPointer::new(String::new());
//! let val = U64Pointer::new(0);
//! let guard = crossbeam_epoch::pin();
//! let cur_ptr_val: &String = ptr.read(&guard);
//!
//! let mut mwcas = MwCas::new();
//! mwcas.compare_exchange(&ptr, cur_ptr_val, String::from("new_string"));
//! mwcas.compare_exchange_u64(&val, 0, 1);
//!
//! assert!(mwcas.exec(&guard));
//! assert_eq!(ptr.read(&guard), &String::from("new_string"));
//! assert_eq!(val.read(&guard), 1);
//! ```
//!
//! # Memory reclamation
//! Drop of values pointed by `HeapPointer` which were replaced by new one during CAS, will
//! be performed by [`crossbeam_epoch`] memory reclamation.  

use crossbeam_epoch::Guard;
use std::borrow::Borrow;
use std::marker::PhantomData;
use std::mem::{align_of_val, size_of};
use std::ops::Deref;
use std::option::Option::Some;
use std::ptr;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};

const STATUS_PREPARE: u8 = 0;
const STATUS_COMPLETED: u8 = 1;
const STATUS_FAILED: u8 = 2;

/// Pointer to data located on heap.
///
/// # Drop
/// Heap memory reference by `HeapPointer` will be released(and structure will be dropped) as part
/// of `HeapPointer` drop.
#[derive(Debug)]
#[repr(transparent)]
pub struct HeapPointer<T> {
    ptr: AtomicU64,
    phantom: PhantomData<T>,
}

impl<T> HeapPointer<T> {
    /// Create new `HeapPointer` which allocates memory for `val` on heap.
    #[inline]
    pub fn new(val: T) -> Self {
        let val_address = Box::into_raw(Box::new(val)) as u64;
        HeapPointer {
            ptr: AtomicU64::new(val_address),
            phantom: PhantomData {},
        }
    }

    /// Read current value of `HeapPointer` and return reference to it.
    #[inline]
    pub fn read<'g>(&'g self, guard: &'g Guard) -> &'g T {
        unsafe { &*self.read_ptr(guard) }
    }

    /// Read current value of `HeapPointer` and return mutable reference to it.
    #[inline]
    pub fn read_mut<'g>(&'g mut self, guard: &'g Guard) -> &'g mut T {
        unsafe { &mut *self.read_ptr(guard) }
    }

    #[inline]
    fn read_ptr(&self, guard: &Guard) -> *mut T {
        read_val(&self.ptr, guard) as *mut u8 as *mut T
    }
}

#[inline]
fn read_val(ptr: &AtomicU64, guard: &Guard) -> u64 {
    loop {
        let cur_val = ptr.load(Ordering::Acquire);
        if let Some(mwcas_ptr) = MwCasPointer::from_poisoned(cur_val, guard) {
            mwcas_ptr.exec_internal(guard);
        } else {
            return cur_val;
        }
    }
}

impl<T: Clone> Clone for HeapPointer<T> {
    fn clone(&self) -> Self {
        let val = self.read(&crossbeam_epoch::pin()).clone();
        HeapPointer::new(val)
    }
}

impl<T> Drop for HeapPointer<T> {
    fn drop(&mut self) {
        unsafe {
            drop(Box::from_raw(
                // this heap pointer cannot be part of any running MwCAS,
                // we can safely use crossbeam_epoch::unprotected()
                self.read_ptr(crossbeam_epoch::unprotected()),
            ));
        }
    }
}

unsafe impl<T: Send> Send for HeapPointer<T> {}
unsafe impl<T: Sync> Sync for HeapPointer<T> {}

/// Pointer to `u64` data.
///
/// This structure is more 'holder' of `u64` than 'pointer'.
/// It exists only to provide interface which is consistent with `HeapPointer` and  
/// can get safe access to current value of `u64` data.
#[derive(Debug)]
#[repr(transparent)]
pub struct U64Pointer {
    val: AtomicU64,
}

impl U64Pointer {
    /// Create new `U64Pointer` with initial value.  
    #[inline]
    pub fn new(val: u64) -> Self {
        Self {
            val: AtomicU64::new(val),
        }
    }

    /// Read current value of pointer.
    #[inline]
    pub fn read(&self, guard: &Guard) -> u64 {
        read_val(&self.val, guard)
    }
}

impl Clone for U64Pointer {
    fn clone(&self) -> Self {
        U64Pointer::new(self.read(&crossbeam_epoch::pin()))
    }
}

unsafe impl Send for U64Pointer {}
unsafe impl Sync for U64Pointer {}

/// Multi-word CAS structure.
///
/// [`MwCas`] contains multi-word CAS state, including pointers which should be changed,
/// original and new pointer values.
/// [`MwCas`] provides `compare and exchange` operations to register CAS operations on pointers.
/// When all `compare and exchange` operations registered, caller should execute `exec` method to
/// actually perform multi-word CAS.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub struct MwCas<'g> {
    // allocated on heap to be safely accessible by 'assisting' threads.
    inner: Box<MwCasInner<'g>>,
    // is MwCAS completed successfully. Used during MwCAS drop.
    success: AtomicBool,
    // Rc used to make this type !Send and !Sync,
    phantom: PhantomData<Rc<u8>>,
}

impl<'g> Default for MwCas<'g> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'g> MwCas<'g> {
    /// Create new `MwCAS`.
    #[inline]
    pub fn new() -> Self {
        MwCas {
            inner: Box::new(MwCasInner {
                status: AtomicU8::new(STATUS_PREPARE),
                cas_ops: Vec::with_capacity(2),
            }),
            success: AtomicBool::new(false),
            phantom: PhantomData {},
        }
    }

    /// Add compare-exchange operation to MwCAS for heap allocated data.
    ///
    /// - `target` points to heap allocated data which should be replaced by new value.  
    /// - `orig_val` is value is from `target` pointer at some point in time, using
    /// `HeapPointer.read()` method.
    /// - `new_val` will be installed to `target` on `MwCas` success. If `MwCas` will fail, then
    /// `new_val` will be dropped.
    #[inline]
    pub fn compare_exchange<T>(&mut self, target: &'g HeapPointer<T>, orig_val: &'g T, new_val: T) {
        #[cfg(debug_assertions)]
        {
            for cas in &self.inner.cas_ops {
                if ptr::eq(cas.target_ptr, &target.ptr as *const AtomicU64) {
                    panic!(
                        "MwCAS cannot compare-and-swap the same {} several times in one execution. 
                        Remove duplicate target reference passed to 'add/with' method. 
                        This can happen if you use unsafe code which skips borrowing rules 
                        checker of Rust: target parameter declared as mutable reference and 
                        cannot be added twice to MwCAS by 'safe' code.",
                        std::any::type_name::<HeapPointer<T>>()
                    )
                }
            }
        }
        let orig_val_ptr = orig_val as *const T as *mut T;
        let orig_val_addr = orig_val_ptr as u64;
        let new_val_ptr = Box::into_raw(Box::new(new_val));
        let new_val_addr = new_val_ptr as u64;
        let drop_fn: Box<dyn Fn(bool) + 'g> = Box::new(move |success| {
            if success {
                drop(unsafe { Box::from_raw(orig_val_ptr) })
            } else {
                drop(unsafe { Box::from_raw(new_val_ptr) })
            }
        });
        self.inner.cas_ops.push(Cas::new(
            &target.ptr as *const AtomicU64 as *mut AtomicU64,
            orig_val_addr,
            new_val_addr,
            drop_fn,
        ));
    }

    /// Add compare-exchange operation to MwCAS for simple u64.
    ///
    /// - `target` struct contains u64 which should be replaced by `MwCas`.  
    /// - `orig_val` is expected value of `target` during CAS.
    /// - `new_val` will be installed to `target` on `MwCas` success.
    #[inline]
    pub fn compare_exchange_u64(&mut self, target: &'g U64Pointer, orig_val: u64, new_val: u64) {
        #[cfg(debug_assertions)]
        {
            for cas in &self.inner.cas_ops {
                if ptr::eq(cas.target_ptr, &target.val as *const AtomicU64) {
                    panic!(
                        "MwCAS cannot compare-and-swap the same {} several times in one execution. 
                        Remove duplicate target reference passed to 'add/with' method. 
                        This can happen if you use unsafe code which skips borrowing rules 
                        checker of Rust: target parameter declared as mutable reference and 
                        cannot be added twice to MwCAS by 'safe' code.",
                        std::any::type_name::<U64Pointer>()
                    )
                }
            }
        }

        let drop_fn: Box<dyn Fn(bool) + 'g> = Box::new(move |_| {});
        self.inner.cas_ops.push(Cas::new(
            &target.val as *const AtomicU64 as *mut AtomicU64,
            *orig_val.borrow(),
            *new_val.borrow(),
            drop_fn,
        ));
    }

    /// Execute all registered CAS operations and return result status.
    ///
    /// `guard` is used for reclamation of memory used by previous values
    /// which were replaced during `MwCas` by new one.
    #[inline]
    pub fn exec(self, guard: &Guard) -> bool {
        let successful_cas = self.inner.exec_internal(guard);
        // delay drop of MwCAS until all thread which can assist to it,
        // e.g. can access this MwCAS by pointer.
        self.success.store(successful_cas, Ordering::Release);
        unsafe {
            guard.defer_unchecked(move || {
                drop(self);
            });
        }
        successful_cas
    }
}

impl<'g> Drop for MwCas<'g> {
    fn drop(&mut self) {
        // if CAS was successful, free memory used by previous value(e.g., value which
        // was replaced). Otherwise, free memory used by 'candidate' value which not
        // used anymore and never will be seen by other threads.
        for cas in &self.inner.cas_ops {
            (cas.drop_fn)(self.success.load(Ordering::Acquire));
        }
    }
}

struct MwCasInner<'g> {
    // MwCAS status(described by const values)
    status: AtomicU8,
    // list of registered CAS operations
    cas_ops: Vec<Cas<'g>>,
}

impl<'g> MwCasInner<'g> {
    #[inline(always)]
    fn status(&self) -> u8 {
        self.status.load(Ordering::Acquire)
    }

    #[inline]
    fn exec_internal(&self, guard: &Guard) -> bool {
        let phase_one_status = self.phase_one(guard);
        let phase_two_status = self.update_status(phase_one_status);
        match phase_two_status {
            Ok(status) => self.phase_two(status),
            Err(cur_status) => {
                self.phase_two(cur_status);
            }
        }
        phase_two_status.map_or_else(|status| status, |status| status) == STATUS_COMPLETED
    }

    /// Phase 1 according to paper
    fn phase_one(&self, guard: &Guard) -> u8 {
        for cas in &self.cas_ops {
            loop {
                match cas.prepare(self, guard) {
                    CasPrepareResult::Conflict(mwcas_ptr) => {
                        if &mwcas_ptr != self.deref() {
                            // we must to try to complete other MWCAS to assists to other thread
                            mwcas_ptr.exec_internal(guard);
                        } else {
                            // if we found our MwCAS => proceed, this indicate that other thread
                            // already assist us.
                            break;
                        }
                    }
                    CasPrepareResult::Success => break,
                    CasPrepareResult::Failed => return STATUS_FAILED,
                }
            }
        }
        STATUS_COMPLETED
    }

    #[inline]
    fn update_status(&self, new_status: u8) -> Result<u8, u8> {
        if let Err(prev_status) = self.status.compare_exchange(
            STATUS_PREPARE,
            new_status,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            // if some other thread executed our MwCAS before us,
            // it already update status and we revert all changes.
            // otherwise, we can overwrite results of completed MwCAS.
            // Description from paper:
            // Installation of a descriptor for a completed PMwCAS (p1) that might
            // inadvertently overwrite the result of another PMwCAS (p2), where
            // p2 should occur after p1. This can happen if a thread T executing p1
            // is about to install a descriptor in a target address A over an existing
            // value V, but goes to sleep. While T sleeps, another thread may complete p1
            // (given the cooperative nature of PMwCAS ) and subsequently
            // p2 executes to set a back to V. If T were to wake up and try to
            // overwrite V (the value it expects) in address A, it would actually be
            // overwriting the result of p2, violating the linearizable schedule for
            // updates to A.
            Err(prev_status)
        } else {
            Ok(new_status)
        }
    }

    /// Phase 2 according to paper
    fn phase_two(&self, mwcas_status: u8) {
        // in any case(success or failure), we should complete CAS
        // on each pointer to obtain a consistent state.
        let mwcas_ptr = MwCasPointer::from(self.deref());
        for cas in &self.cas_ops {
            cas.complete(mwcas_status, &mwcas_ptr);
        }
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
struct MwCasPointer<'g> {
    mwcas: &'g MwCasInner<'g>,
}

impl<'g> Deref for MwCasPointer<'g> {
    type Target = MwCasInner<'g>;

    fn deref(&self) -> &Self::Target {
        self.mwcas
    }
}

impl<'g> MwCasPointer<'g> {
    const MWCAS_FLAG: u64 = 0x4000_0000_0000_0000;

    /// Try to create pointer to existing `MwCAS` based on address installed on CAS target pointer.
    #[inline]
    fn from_poisoned(poisoned_addr: u64, _: &'g Guard) -> Option<MwCasPointer<'g>> {
        let valid_addr = poisoned_addr & !Self::MWCAS_FLAG;
        if poisoned_addr != valid_addr {
            Option::Some(MwCasPointer {
                // we observe existing MwCas during of guard lifetime
                // it's safe to access it until guard is alive
                mwcas: unsafe { &*(valid_addr as *const u64 as *const MwCasInner) },
            })
        } else {
            // passed address is not `poisoned` address,
            // e.g. not an address of some existing `MwCAS`
            Option::None
        }
    }

    /// Return address of MwCas structure but with modified high bits which
    /// indicate that this address is not valid address of MwCas structure
    /// but a special pointer to MwCas.
    #[inline(always)]
    fn poisoned(&self) -> u64 {
        let addr = self.mwcas as *const MwCasInner as *const u64 as u64;
        addr | Self::MWCAS_FLAG
    }
}

impl<'g> From<&'g MwCasInner<'g>> for MwCasPointer<'g> {
    fn from(mwcas: &'g MwCasInner) -> Self {
        MwCasPointer { mwcas }
    }
}

impl<'g> Eq for MwCasPointer<'g> {}

impl<'g> PartialEq for MwCasPointer<'g> {
    fn eq(&self, other: &MwCasPointer) -> bool {
        ptr::eq(self.mwcas, other.mwcas)
    }
}

impl<'g> PartialEq<MwCasInner<'g>> for MwCasPointer<'g> {
    fn eq(&self, other: &MwCasInner) -> bool {
        ptr::eq(self.mwcas, other)
    }
}

/// Struct describe one CAS operation of `MwCAS`.
struct Cas<'g> {
    target_ptr: *mut AtomicU64,
    orig_val: u64,
    new_val: u64,
    // function which will drop original/new value after CAS completion
    drop_fn: Box<dyn Fn(bool) + 'g>,
}

unsafe impl<'g> Send for Cas<'g> {}
unsafe impl<'g> Sync for Cas<'g> {}

#[derive(PartialEq, Copy, Clone)]
enum CasPrepareResult<'g> {
    Success,
    Conflict(MwCasPointer<'g>),
    Failed,
}

impl<'g> Cas<'g> {
    fn new(
        pointer: *mut AtomicU64,
        orig_val: u64,
        new_val: u64,
        drop_fn: Box<dyn Fn(bool) + 'g>,
    ) -> Self {
        let max_addr: u64 = 0xDFFF_FFFF_FFFF_FFFF;
        assert!(!pointer.is_null(), "Pointer must be non null");
        debug_assert!(
            (pointer as u64) < max_addr,
            "Pointer must point to memory in range [0x{:X}, 0x{:X}], because MwCas \
             use highest 3 bits of address for internal use. Actual address to which pointer \
             points was 0x{:x}",
            0,
            max_addr,
            pointer as u64
        );
        unsafe {
            let align = align_of_val(&*pointer);
            debug_assert_eq!(
                align,
                size_of::<u64>(),
                "Pointer must be align on {} bytes, but pointer was aligned on {}",
                size_of::<u64>(),
                align
            )
        }
        debug_assert!(
            orig_val < MwCasPointer::MWCAS_FLAG,
            "MwCas can be applied only for original values < {}. Actual value was {}",
            MwCasPointer::MWCAS_FLAG,
            orig_val
        );
        debug_assert!(
            new_val < MwCasPointer::MWCAS_FLAG,
            "MwCas can be applied only for new values < {}. Actual value was {}",
            MwCasPointer::MWCAS_FLAG,
            new_val
        );

        Cas {
            target_ptr: pointer,
            orig_val,
            new_val,
            drop_fn,
        }
    }

    /// Try to install pointer to `MwCAS` into value of current CAS target.
    fn prepare<'a>(&self, mwcas: &MwCasInner, guard: &'a Guard) -> CasPrepareResult<'a> {
        let new_val = MwCasPointer::from(mwcas.deref()).poisoned();
        let prev = unsafe {
            (*self.target_ptr)
                .compare_exchange(self.orig_val, new_val, Ordering::AcqRel, Ordering::Acquire)
                .map_or_else(|v| v, |v| v)
        };

        if prev == self.orig_val {
            CasPrepareResult::Success
        } else if let Some(mwcas_ptr) = MwCasPointer::from_poisoned(prev, guard) {
            // found MWCAS pointer installed by some other
            CasPrepareResult::Conflict(mwcas_ptr)
        } else {
            CasPrepareResult::Failed
        }
    }

    /// Complete CAS operation for current pointer: set new value on MwCAS success or rollback to
    /// original value if MwCAS failed.
    fn complete(&self, status: u8, mwcas: &MwCasPointer) {
        let new_val = match status {
            STATUS_COMPLETED => self.new_val,
            STATUS_FAILED => self.orig_val,
            _ => panic!("CAS cannot be completed for not prepared MWCAS"),
        };
        let expected_val = mwcas.poisoned();
        unsafe {
            let _ = (*self.target_ptr).compare_exchange(
                expected_val,
                new_val,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        };
        // if CAS above failed, then some other thread completed our MwCAS,
        // e.g assist us. This is expected case, no additional actions required.
        // Or we found MwCas of installed by other thread. This is also expected
        // case when we fail our MwCAS and some other MwCas install it pointer to same memory cell.
    }
}

#[cfg(test)]
mod tests {
    use crate::Cas;
    use std::sync::atomic::Ordering;

    mod simple {
        use crate::{HeapPointer, MwCas, U64Pointer, STATUS_COMPLETED, STATUS_FAILED};
        use std::ops::Deref;
        use std::ptr::NonNull;
        use std::sync::atomic::Ordering;

        #[test]
        fn test_mwcas_add_ptr() {
            let guard = crossbeam_epoch::pin();
            let val1 = HeapPointer::new(5);
            let val2 = HeapPointer::new(10);
            let val3 = U64Pointer::new(15);
            let new_val1 = 15;
            let new_val2 = 20;
            let new_val3 = 25;
            let orig_val1 = val1.read(&guard);
            let orig_val2 = val2.read(&guard);
            let orig_val3 = val3.read(&guard);

            let mut mw_cas = MwCas::new();
            mw_cas.compare_exchange(&val1, orig_val1, new_val1);
            mw_cas.compare_exchange(&val2, orig_val2, new_val2);
            mw_cas.compare_exchange_u64(&val3, orig_val3, new_val3);
            assert!(mw_cas.exec(&guard));
            assert_eq!(*val1.read(&guard), new_val1);
            assert_eq!(*val2.read(&guard), new_val2);
            assert_eq!(val3.read(&guard), new_val3);
        }

        #[test]
        #[should_panic]
        fn test_add_same_ptr() {
            let guard = crossbeam_epoch::pin();
            let val1 = HeapPointer::new(5);
            let new_val1 = 15;
            let orig_val1 = val1.read(&guard);

            let mut mw_cas = MwCas::new();
            mw_cas.compare_exchange(&val1, orig_val1, new_val1);
            mw_cas.compare_exchange(&val1, orig_val1, new_val1);
        }

        #[test]
        #[should_panic]
        fn test_add_same_u64_val() {
            let guard = crossbeam_epoch::pin();
            let val1 = U64Pointer::new(5);
            let new_val1 = 15;
            let orig_val1 = val1.read(&guard);

            let mut mw_cas = MwCas::new();
            mw_cas.compare_exchange_u64(&val1, orig_val1, new_val1);
            mw_cas.compare_exchange_u64(&val1, orig_val1, new_val1);
        }

        #[test]
        fn test_prepared_cas_completion_assist() {
            let val1 = HeapPointer::new(1);
            let val2 = HeapPointer::new(2);
            let guard = crossbeam_epoch::pin();
            let orig_val1 = val1.read(&guard);
            let orig_val2 = val2.read(&guard);
            let mut mwcas = MwCas::new();
            mwcas.compare_exchange(&val1, orig_val1, 2);
            mwcas.compare_exchange(&val2, orig_val2, 3);

            // emulate that some other thread begins our MwCAS
            let cas1 = mwcas.inner.cas_ops.first().unwrap();
            let cas2 = mwcas.inner.cas_ops.get(1).unwrap();
            cas1.prepare(mwcas.inner.deref(), &guard);
            cas2.prepare(mwcas.inner.deref(), &guard);

            assert!(mwcas.exec(&guard));
            assert_eq!(*val1.read(&guard), 2);
            assert_eq!(*val2.read(&guard), 3);

            let orig_val1 = val1.read(&guard);
            let orig_val2 = val2.read(&guard);
            let mut mwcas = MwCas::new();
            mwcas.compare_exchange(&val1, orig_val1, 3);
            mwcas.compare_exchange(&val2, orig_val2, 4);
            // emulate that some other thread begins our MwCAS
            let cas1 = mwcas.inner.cas_ops.last().unwrap();
            cas1.prepare(mwcas.inner.deref(), &guard);

            assert!(mwcas.exec(&guard));
            assert_eq!(*val1.read(&guard), 3);
            assert_eq!(*val2.read(&guard), 4);
        }

        #[test]
        fn test_cas_completion_assist_on_subset_of_references() {
            let val1 = HeapPointer::new(1);
            let val2 = HeapPointer::new(2);
            let val3 = HeapPointer::new(3);
            let guard = crossbeam_epoch::pin();
            let mut mwcas1 = MwCas::new();
            let mut mwcas2 = MwCas::new();
            let orig_val1 = val1.read(&guard);
            let orig_val2 = val2.read(&guard);
            let orig_val3 = val3.read(&guard);
            mwcas1.compare_exchange(&val1, orig_val1, 2);
            mwcas1.compare_exchange(&val2, orig_val2, 3);
            mwcas2.compare_exchange(&val3, orig_val3, 4);

            // assist first MwCAS
            let cas1 = mwcas1.inner.cas_ops.first().unwrap();
            cas1.prepare(mwcas1.inner.deref(), &guard);

            // at start, second MwCAS should complete first MwCAS
            // and then can successfully complete it's own operations.
            assert!(mwcas2.exec(&guard));
            assert_eq!(*val3.read(&guard), 4);
            assert!(mwcas1.exec(&guard));
            assert_eq!(*val1.read(&guard), 2);
            assert_eq!(*val2.read(&guard), 3);
        }

        #[test]
        fn test_assist_not_change_cas_result() {
            let mut val1 = HeapPointer::new(1);
            let value1 = unsafe { NonNull::new_unchecked(&mut val1) };
            let mut val2 = HeapPointer::new(2);
            let value2 = unsafe { NonNull::new_unchecked(&mut val2) };
            let guard = crossbeam_epoch::pin();
            let mut mwcas1 = MwCas::new();
            let mut mwcas2 = MwCas::new();
            let val1_ref = val1.read(&guard);
            unsafe {
                mwcas1.compare_exchange(&*value1.as_ptr(), val1_ref, 2);
                mwcas1.compare_exchange(&*value2.as_ptr(), val1_ref, 2);
            }
            assert_eq!(mwcas1.inner.phase_one(&guard), STATUS_FAILED);
            mwcas1.inner.update_status(STATUS_FAILED).unwrap();

            // this cause assist to mwcas-1 which already on fail path
            unsafe {
                mwcas2.compare_exchange(&*value1.as_ptr(), val1_ref, 2);
            }
            assert!(mwcas2.exec(&guard));
            assert_eq!(mwcas1.inner.status(), STATUS_FAILED);
            assert!(!mwcas1.exec(&guard));

            assert_eq!(*val1.read(&guard), 2);
            assert_eq!(*val2.read(&guard), 2);
        }

        #[test]
        #[ignore]
        fn test_mwcas_race_in_phase_one_before_status_update() {
            let mut val1 = HeapPointer::new(1);
            let value1 = unsafe { NonNull::new_unchecked(&mut val1) };
            let mut val2 = HeapPointer::new(2);
            let value2 = unsafe { NonNull::new_unchecked(&mut val2) };
            let mut val3 = HeapPointer::new(3);
            let value3 = unsafe { NonNull::new_unchecked(&mut val3) };
            let guard = crossbeam_epoch::pin();
            let mut mwcas1 = MwCas::new();
            let mut mwcas2 = MwCas::new();
            unsafe {
                mwcas1.compare_exchange(&*value1.as_ptr(), val1.read(&guard), 2);
                mwcas1.compare_exchange(&*value2.as_ptr(), val2.read(&guard), 3);
                mwcas2.compare_exchange(&*value3.as_ptr(), val3.read(&guard), 4);
            }

            // start phase 1 of 1st mwcas
            let status = mwcas1.inner.phase_one(&guard);
            assert_eq!(status, STATUS_COMPLETED);
            // execute 2nd mwcas which should find conflicting 1st MwCAS in value2,
            // assist it and complete both MwCASs
            assert!(mwcas2.exec(&guard));
            assert_eq!(*val1.read(&guard), 2);
            assert_eq!(*val2.read(&guard), 4);
            assert_eq!(*val3.read(&guard), 4);
            // execute phase 2 for completed MwCas and check that result remains the same
            mwcas1.inner.phase_two(STATUS_COMPLETED);
            assert_eq!(*val1.read(&guard), 1);
            assert_eq!(*val2.read(&guard), 4);
            assert_eq!(*val3.read(&guard), 4);
            mwcas1.success.store(true, Ordering::Release);
        }

        #[test]
        #[ignore]
        fn test_mwcas_race_in_phase_one_after_status_update() {
            let mut mwcas1 = MwCas::new();
            let mut mwcas2 = MwCas::new();

            let mut val1 = HeapPointer::new(1);
            let value1 = unsafe { NonNull::new_unchecked(&mut val1) };
            let mut val2 = HeapPointer::new(2);
            let value2 = unsafe { NonNull::new_unchecked(&mut val2) };
            unsafe {
                mwcas1.compare_exchange(&*value1.as_ptr(), &1, 2);
                mwcas1.compare_exchange(&*value2.as_ptr(), &2, 3);
                mwcas2.compare_exchange(&*value2.as_ptr(), &3, 4);
            }

            let guard = crossbeam_epoch::pin();
            // start phase 1 of 1st mwcas
            let status = mwcas1.inner.phase_one(&guard);
            mwcas1.inner.update_status(status).unwrap();
            // execute 2nd mwcas which should find conflicting 1st MwCAS in value2,
            // assist it and complete both MwCASs
            mwcas2.exec(&guard);
            assert_eq!(*val1.read(&guard), 2);
            assert_eq!(*val2.read(&guard), 3);
            // execute phase 2 for completed MwCas and check that result remains the same
            mwcas1.inner.phase_two(status);
            assert_eq!(*val1.read(&guard), 2);
            assert_eq!(*val2.read(&guard), 3);
        }

        #[test]
        #[ignore]
        fn test_mwcas_fail_when_concurrent_mwcas_won_race() {
            let mut val1 = HeapPointer::new(1);
            let mut val2 = HeapPointer::new(2);
            let value1 = unsafe { NonNull::new_unchecked(&mut val1) };
            let value2 = unsafe { NonNull::new_unchecked(&mut val2) };
            let guard = crossbeam_epoch::pin();
            let mut mwcas1 = MwCas::new();
            let mut mwcas2 = MwCas::new();
            unsafe {
                mwcas1.compare_exchange(&*value1.as_ptr(), val1.read(&guard), 2);
                mwcas1.compare_exchange(&*value2.as_ptr(), val2.read(&guard), 3);
                // emulate race with 2nd MwCAS on same value
                mwcas2.compare_exchange(&*value2.as_ptr(), val2.read(&guard), 4);
            }

            let cas = mwcas1.inner.cas_ops.first().unwrap();
            // emulate that only 1 CAS started in 1st MwCAS
            cas.prepare(mwcas1.inner.deref(), &guard);

            mwcas2.exec(&guard);
            assert_eq!(*val2.read(&guard), 4);

            // try complete 1st MwCAS which should fail because 2nd already
            // update expected field value
            assert!(!mwcas1.exec(&guard));
        }

        #[test]
        #[ignore]
        fn test_mwcas_linearization() {
            let mut mwcas1 = MwCas::new();
            let mut mwcas2 = MwCas::new();

            let mut val1 = HeapPointer::new(1);
            let value1 = unsafe { NonNull::new_unchecked(&mut val1) };
            let mut val2 = HeapPointer::new(2);
            let value2 = unsafe { NonNull::new_unchecked(&mut val2) };
            unsafe {
                mwcas1.compare_exchange(&*value1.as_ptr(), &1, 2);
                mwcas1.compare_exchange(&*value2.as_ptr(), &2, 3);
                mwcas2.compare_exchange(&*value1.as_ptr(), &2, 1);
                mwcas2.compare_exchange(&*value2.as_ptr(), &3, 2);
            }

            let guard = crossbeam_epoch::pin();
            // emulate start of 1st MwCAS without status update
            mwcas1.inner.phase_one(&guard);

            // 2nd MwCAS will assist to 1st MwCAS, complete itself(rollback
            // all fields to original values)
            assert!(mwcas2.exec(&guard));
            // 1st MwCAS should skip all field updates because someone already done it's work
            // and revert field values back
            assert!(mwcas1.exec(&guard));

            assert_eq!(*val1.read(&guard), 1);
            assert_eq!(*val2.read(&guard), 2);
        }

        #[test]
        fn test_mwcas_completion_on_pointer_read() {
            let mut val = HeapPointer::new(1);
            let value = unsafe { NonNull::new_unchecked(&mut val) };
            let guard = crossbeam_epoch::pin();
            let mut mwcas = MwCas::new();
            unsafe {
                mwcas.compare_exchange(&*value.as_ptr(), val.read(&guard), 2);
            }

            assert_eq!(*val.read(&guard), 1);
            assert_eq!(mwcas.inner.phase_one(&guard), STATUS_COMPLETED);
            assert_eq!(*val.read(&guard), 2);
            mwcas.success.store(true, Ordering::Release);
        }
    }

    impl<'g> Cas<'g> {
        #[inline]
        fn current_value(&self) -> u64 {
            unsafe { (*self.target_ptr).load(Ordering::Acquire) }
        }
    }

    mod mwcas_pointer_test {
        use crate::{MwCas, MwCasPointer};
        use std::ops::Deref;
        use std::ptr;

        #[test]
        fn create_pointer_from_structure() {
            let mw_cas = MwCas::new();
            let ptr = MwCasPointer::from(mw_cas.inner.deref());
            assert!(ptr::eq(ptr.deref(), mw_cas.inner.deref()));
            let guard = crossbeam_epoch::pin();
            assert!(matches!(
                MwCasPointer::from_poisoned(ptr.poisoned(), &guard),
                Some(_)
            ));
        }

        #[test]
        fn create_pointer_from_address() {
            let guard = crossbeam_epoch::pin();
            let mw_cas = MwCas::new();
            let parsed_ptr = MwCasPointer::from_poisoned(
                MwCasPointer::from(mw_cas.inner.deref()).poisoned(),
                &guard,
            );
            assert!(parsed_ptr.is_some());
            let ptr = parsed_ptr.unwrap();
            assert!(ptr::eq(ptr.deref(), mw_cas.inner.deref()));

            assert_eq!(
                ptr.poisoned(),
                MwCasPointer::from(mw_cas.inner.deref()).poisoned()
            );
        }

        #[test]
        fn create_pointer_from_invalid_address() {
            let mw_cas = MwCas::new();
            let addr = &mw_cas as *const MwCas as u64;
            let guard = crossbeam_epoch::pin();
            let parsed_ptr = MwCasPointer::from_poisoned(addr, &guard);
            assert!(parsed_ptr.is_none());
        }
    }

    mod cas_tests {
        use crate::{
            CasPrepareResult, HeapPointer, MwCas, MwCasPointer, STATUS_COMPLETED, STATUS_FAILED,
        };
        use std::ops::Deref;
        use std::sync::atomic::Ordering;

        #[test]
        fn test_cas_success_completion() {
            let guard = crossbeam_epoch::pin();
            let cur_val = HeapPointer::new(1);
            let mut mwcas = MwCas::new();
            let orig_val = cur_val.read(&guard);
            mwcas.compare_exchange(&cur_val, orig_val, 2);
            let cas = mwcas.inner.cas_ops.first().unwrap();

            assert!(matches!(
                cas.prepare(mwcas.inner.deref(), &guard),
                CasPrepareResult::Success
            ));

            let mwcas_ptr = MwCasPointer::from(mwcas.inner.deref());
            assert!(
                matches!(MwCasPointer::from_poisoned(cas.current_value(), &guard),
                    Some(ptr) if mwcas_ptr == ptr)
            );

            cas.complete(STATUS_COMPLETED, &mwcas_ptr);
            mwcas.success.store(true, Ordering::Release);
            assert_eq!(*cur_val.read(&guard), 2);
        }

        #[test]
        fn test_complete_cas_with_failure() {
            let guard = crossbeam_epoch::pin();
            let value = HeapPointer::new(1);
            let mut mwcas = MwCas::new();
            let orig_val = value.read(&guard);
            mwcas.compare_exchange(&value, orig_val, 2);
            let cas = mwcas.inner.cas_ops.first().unwrap();

            assert!(matches!(
                cas.prepare(mwcas.inner.deref(), &guard),
                CasPrepareResult::Success
            ));
            let mwcas_ptr = MwCasPointer::from(mwcas.inner.deref());
            assert!(
                matches!(MwCasPointer::from_poisoned(cas.current_value(), &guard),
                    Some(ptr) if mwcas_ptr == ptr)
            );

            cas.complete(STATUS_FAILED, &mwcas_ptr);
            mwcas.success.store(false, Ordering::Release);
            assert_eq!(*value.read(&guard), 1);
        }

        #[test]
        fn test_same_cas_conflict() {
            let guard = crossbeam_epoch::pin();
            let val1 = HeapPointer::new(1);
            let mut mwcas = MwCas::new();
            let orig_val = val1.read(&guard);
            mwcas.compare_exchange(&val1, orig_val, 2);
            let cas = mwcas.inner.cas_ops.first().unwrap();
            let mwcas_ptr = MwCasPointer::from(mwcas.inner.deref());
            assert!(matches!(
                cas.prepare(mwcas.inner.deref(), &guard),
                CasPrepareResult::Success
            ));
            assert!(matches!(
                cas.prepare(mwcas.inner.deref(), &guard),
                CasPrepareResult::Conflict(ptr) if ptr == mwcas_ptr
            ));
            cas.complete(STATUS_COMPLETED, &mwcas_ptr);
            mwcas.success.store(true, Ordering::Release);
        }

        #[test]
        #[should_panic]
        fn test_cas_completion_with_invalid_status() {
            let mut value = HeapPointer::new(1);
            let mut mwcas = MwCas::new();
            mwcas.compare_exchange(&value, &1, 2);
            let cas = mwcas.inner.cas_ops.first().unwrap();
            cas.complete(u8::MAX, &MwCasPointer::from(mwcas.inner.deref()));
        }
    }
}
