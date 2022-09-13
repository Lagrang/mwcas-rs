use crossbeam_epoch::Guard;
use crossbeam_utils::atomic::AtomicCell;
use crossbeam_utils::thread;
use mwcas::{HeapPointer, MwCas, U64Pointer};
use rand::distributions::Uniform;
use rand::prelude::*;
use std::collections::HashMap;
use std::ptr::NonNull;

const CAS_OPS: usize = 80_000;

#[test]
fn test_on_all_logical_threads_with_dataset_contention() {
    run_multi_threaded(num_cpus::get(), num_cpus::get(), CAS_OPS);
    run_multi_threaded(num_cpus::get(), num_cpus::get() * 2, CAS_OPS);
    run_multi_threaded(num_cpus::get(), num_cpus::get() * 4, CAS_OPS);
    run_multi_threaded(num_cpus::get(), num_cpus::get() * 8, CAS_OPS);
}

#[test]
fn test_on_all_logical_threads_no_dataset_contention() {
    run_multi_threaded(num_cpus::get(), num_cpus::get() * 50, CAS_OPS);
    run_multi_threaded(num_cpus::get(), num_cpus::get() * 100, CAS_OPS);
}

#[test]
fn test_on_doubled_logical_threads_no_dataset_contention() {
    run_multi_threaded(num_cpus::get() * 2, num_cpus::get() * 50, CAS_OPS);
    run_multi_threaded(num_cpus::get() * 2, num_cpus::get() * 100, CAS_OPS);
}

#[test]
fn test_on_many_logical_threads_no_dataset_contention() {
    run_multi_threaded(num_cpus::get() * 4, num_cpus::get() * 50, CAS_OPS);
    run_multi_threaded(num_cpus::get() * 8, num_cpus::get() * 50, CAS_OPS);
    run_multi_threaded(num_cpus::get() * 4, num_cpus::get() * 100, CAS_OPS);
    run_multi_threaded(num_cpus::get() * 8, num_cpus::get() * 100, CAS_OPS);
}

#[test]
fn test_on_doubled_logical_threads_with_dataset_contention() {
    run_multi_threaded(num_cpus::get() * 2, num_cpus::get(), CAS_OPS);
    run_multi_threaded(num_cpus::get() * 2, num_cpus::get() * 2, CAS_OPS);
    run_multi_threaded(num_cpus::get() * 2, num_cpus::get() * 4, CAS_OPS);
}

#[test]
fn test_on_many_logical_threads_with_dataset_contention() {
    run_multi_threaded(num_cpus::get() * 4, num_cpus::get(), CAS_OPS);
    run_multi_threaded(num_cpus::get() * 4, num_cpus::get() * 2, CAS_OPS);
    run_multi_threaded(num_cpus::get() * 4, num_cpus::get() * 4, CAS_OPS);
    run_multi_threaded(num_cpus::get() * 8, num_cpus::get(), CAS_OPS);
    run_multi_threaded(num_cpus::get() * 8, num_cpus::get() * 2, CAS_OPS);
    run_multi_threaded(num_cpus::get() * 8, num_cpus::get() * 4, CAS_OPS);
}

fn run_multi_threaded(threads: usize, data_set_size: usize, cas_ops_count: usize) {
    let mut data_set: Vec<CasData> = Vec::with_capacity(data_set_size);
    for _ in 0..data_set.capacity() {
        data_set.push(CasData::new());
    }
    let uniform = Uniform::new(0, data_set.len());
    let data_set: AtomicCell<&mut Vec<CasData>> = AtomicCell::new(&mut data_set);

    let mut vec: Vec<(usize, ChangeSnapshot)> = thread::scope(|s| {
        let mut res = Vec::new();
        for _ in 0..threads {
            let handle = s.spawn(|_| {
                let mut res = Vec::new();
                let mut rng = thread_rng();
                let mut guard = crossbeam_epoch::pin();
                for i in 0..cas_ops_count {
                    if i % 10 == 0 {
                        guard = crossbeam_epoch::pin();
                    }
                    let cas_idx = rng.sample(uniform);
                    let cas_data = unsafe { &mut (*data_set.as_ptr())[cas_idx] };
                    let field1 = unsafe { NonNull::new_unchecked(&mut cas_data.field_1) };
                    let field2 = unsafe { NonNull::new_unchecked(&mut cas_data.field_2) };
                    let field3 = unsafe { NonNull::new_unchecked(&mut cas_data.field_3) };
                    let field4 = unsafe { NonNull::new_unchecked(&mut cas_data.field_4) };
                    let mut cas_change = CasChange::generate_for(cas_data, &guard);
                    let mut mwcas = MwCas::new();
                    mwcas.compare_exchange(
                        unsafe { &mut *field1.as_ptr() },
                        cas_change.field_1_orig_val,
                        cas_change.field_1_new_val,
                    );
                    mwcas.compare_exchange(
                        unsafe { &mut *field2.as_ptr() },
                        cas_change.field_2_orig_val,
                        cas_change.field_2_new_val,
                    );
                    mwcas.compare_exchange(
                        unsafe { &mut *field3.as_ptr() },
                        cas_change.field_3_orig_val,
                        cas_change.field_3_new_val,
                    );
                    mwcas.compare_exchange_u64(
                        unsafe { &mut *field4.as_ptr() },
                        cas_change.field_4_orig_val,
                        cas_change.field_4_new_val,
                    );
                    cas_change.is_completed = mwcas.exec(&guard);
                    res.push((cas_idx, cas_change.to_snapshot()));
                }
                res
            });
            res.push(handle);
        }

        let mut changes = Vec::new();
        for handle in res {
            changes.append(&mut handle.join().unwrap());
        }
        changes
    })
    .unwrap();

    let mut map: HashMap<usize, Vec<ChangeSnapshot>> = HashMap::new();
    for (i, change) in vec.drain(..) {
        map.entry(i).or_default().push(change);
    }

    for (i, changes) in map.drain() {
        let cas_data = unsafe { &(*data_set.as_ptr())[i] };
        check_linearization(cas_data, &changes);
        print_cas_stats(&changes)
    }
}

fn check_linearization(cas: &CasData, changes: &[ChangeSnapshot]) {
    let mut prev_change = &cas.get_initial_change();
    let mut linear_order_found = false;
    let mut next = 1;
    let guard = crossbeam_epoch::pin();
    let field_1 = cas.field_1.read(&guard);
    let field_2 = cas.field_2.read(&guard);
    let field_3 = cas.field_3.read(&guard);
    let field_4 = cas.field_4.read(&guard);

    while next < changes.len() {
        let next_change = &changes[next];
        if !next_change.is_completed {
            next += 1;
            continue;
        }

        if *field_1 == next_change.field_1_new_val
            && *field_2 == next_change.field_2_new_val
            && *field_3 == next_change.field_3_new_val
            && field_4 == next_change.field_4_new_val
        {
            linear_order_found = true;
            break;
        }

        if next_change.is_based_on(prev_change) {
            prev_change = next_change;
            next = 1;
        } else {
            next += 1;
        }
    }
    assert!(
        linear_order_found,
        "CAS operation changes cannot be linearized: chain of CAS operations which leads \
                to latest data state not found"
    );
    println!(
        "Linearization found at {:?} element(elements={:?})",
        next + 1,
        changes.len()
    );
}

fn print_cas_stats(changes: &[ChangeSnapshot]) {
    let mut success_cas_count: usize = 0;
    let mut failed_cas_count: usize = 0;
    for change in changes.iter() {
        if change.is_completed {
            failed_cas_count += 1
        } else {
            success_cas_count += 1
        }
    }
    println!(
        "CAS changes: success: {:?}, failed: {:?}",
        success_cas_count, failed_cas_count
    )
}

unsafe impl Send for CasData {}
unsafe impl Sync for CasData {}

struct CasData {
    field_1: HeapPointer<u64>,
    field_2: HeapPointer<u64>,
    field_3: HeapPointer<u64>,
    field_4: U64Pointer,
}

#[derive(Copy, Clone)]
struct ChangeSnapshot {
    is_completed: bool,
    field_1_orig_val: u64,
    field_1_new_val: u64,
    field_2_orig_val: u64,
    field_2_new_val: u64,
    field_3_orig_val: u64,
    field_3_new_val: u64,
    field_4_orig_val: u64,
    field_4_new_val: u64,
}

impl ChangeSnapshot {
    fn is_based_on(&self, other_change: &ChangeSnapshot) -> bool {
        self.field_1_orig_val == other_change.field_1_new_val
            && self.field_2_orig_val == other_change.field_2_new_val
            && self.field_3_orig_val == other_change.field_3_new_val
            && self.field_4_orig_val == other_change.field_4_new_val
    }
}

#[derive(Copy, Clone)]
struct CasChange<'g> {
    is_completed: bool,
    field_1_orig_val: &'g u64,
    field_1_new_val: u64,
    field_2_orig_val: &'g u64,
    field_2_new_val: u64,
    field_3_orig_val: &'g u64,
    field_3_new_val: u64,
    field_4_orig_val: u64,
    field_4_new_val: u64,
}

impl CasData {
    fn new() -> CasData {
        let field_1 = HeapPointer::new(0);
        let field_2 = HeapPointer::new(0);
        let field_3 = HeapPointer::new(0);
        let field_4 = U64Pointer::new(0);
        CasData {
            field_1,
            field_2,
            field_3,
            field_4,
        }
    }

    fn get_initial_change(&self) -> ChangeSnapshot {
        ChangeSnapshot {
            is_completed: true,
            field_1_orig_val: 0,
            field_1_new_val: 0,
            field_2_orig_val: 0,
            field_2_new_val: 0,
            field_3_orig_val: 0,
            field_3_new_val: 0,
            field_4_orig_val: 0,
            field_4_new_val: 0,
        }
    }
}

impl<'g> CasChange<'g> {
    fn generate_for(cas_data: &'g CasData, guard: &'g Guard) -> CasChange<'g> {
        let field1 = cas_data.field_1.read(guard);
        let field2 = cas_data.field_2.read(guard);
        let field3 = cas_data.field_3.read(guard);
        let field4 = cas_data.field_4.read(guard);
        CasChange {
            is_completed: false,
            field_1_orig_val: field1,
            field_1_new_val: *field1 + 1,
            field_2_orig_val: field2,
            field_2_new_val: *field2 + 1,
            field_3_orig_val: field3,
            field_3_new_val: *field3 + 1,
            field_4_orig_val: field4,
            field_4_new_val: field4 + 1,
        }
    }

    fn to_snapshot(self) -> ChangeSnapshot {
        ChangeSnapshot {
            is_completed: self.is_completed,
            field_1_orig_val: *self.field_1_orig_val,
            field_1_new_val: self.field_1_new_val,
            field_2_orig_val: *self.field_2_orig_val,
            field_2_new_val: self.field_2_new_val,
            field_3_orig_val: *self.field_3_orig_val,
            field_3_new_val: self.field_3_new_val,
            field_4_orig_val: self.field_4_orig_val,
            field_4_new_val: self.field_4_new_val,
        }
    }
}
