use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use crossbeam_epoch::Guard;
use crossbeam_utils::atomic::AtomicCell;
use crossbeam_utils::thread;
use mwcas::{HeapPointer, MwCas, U64Pointer};
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};
use std::ptr::NonNull;
use std::time::Instant;

struct CasData {
    field_1: HeapPointer<u64>,
    field_2: HeapPointer<u64>,
    field_3: U64Pointer,
}

unsafe impl Send for CasData {}
unsafe impl Sync for CasData {}

impl CasData {
    fn new() -> CasData {
        let field_1 = HeapPointer::new(0);
        let field_2 = HeapPointer::new(0);
        let field_3 = U64Pointer::new(0);
        CasData {
            field_1,
            field_2,
            field_3,
        }
    }
}

#[derive(Copy, Clone)]
struct CasChange<'g> {
    field_1_orig_val: &'g u64,
    field_1_new_val: u64,
    field_2_orig_val: &'g u64,
    field_2_new_val: u64,
    field_3_orig_val: u64,
    field_3_new_val: u64,
}

impl<'g> CasChange<'g> {
    fn generate_for(cas_data: &'g CasData, guard: &'g Guard) -> CasChange<'g> {
        let field1 = cas_data.field_1.read(guard);
        let field2 = cas_data.field_2.read(guard);
        let field3 = cas_data.field_3.read(guard);
        CasChange {
            field_1_orig_val: field1,
            field_1_new_val: *field1 + 1,
            field_2_orig_val: field2,
            field_2_new_val: *field2 + 1,
            field_3_orig_val: field3,
            field_3_new_val: field3 + 1,
        }
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent");
    for size in (10_000..=50_000usize).step_by(10_000) {
        group.bench_with_input(BenchmarkId::new("node_size", size), &size, |b, size| {
            let mut data_set: Vec<CasData> = Vec::with_capacity(*size);
            for _ in 0..data_set.capacity() {
                data_set.push(CasData::new());
            }
            let uniform = Uniform::new(0, data_set.len());
            let data_set: AtomicCell<&mut Vec<CasData>> = AtomicCell::new(&mut data_set);

            b.iter_custom(|iters| {
                let start = thread::scope(|s| {
                    for _ in 0..num_cpus::get() {
                        s.spawn(|_| {
                            let mut rng = thread_rng();
                            for _ in 0..iters {
                                let guard = crossbeam_epoch::pin();
                                let cas_idx = rng.sample(uniform);
                                let cas_data = unsafe { &mut (*data_set.as_ptr())[cas_idx] };
                                let field1 =
                                    unsafe { NonNull::new_unchecked(&mut cas_data.field_1) };
                                let field2 =
                                    unsafe { NonNull::new_unchecked(&mut cas_data.field_2) };
                                let field3 =
                                    unsafe { NonNull::new_unchecked(&mut cas_data.field_3) };
                                let cas_change = CasChange::generate_for(cas_data, &guard);
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
                                mwcas.compare_exchange_u64(
                                    unsafe { &mut *field3.as_ptr() },
                                    cas_change.field_3_orig_val,
                                    cas_change.field_3_new_val,
                                );
                                mwcas.exec(&guard);
                            }
                        });
                    }
                    Instant::now()
                })
                .unwrap();

                start.elapsed()
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group("single_threaded");
    group.bench_function("three_cas", |b| {
        b.iter(|| {
            let mut cas_data = CasData::new();
            let field1 = unsafe { NonNull::new_unchecked(&mut cas_data.field_1) };
            let field2 = unsafe { NonNull::new_unchecked(&mut cas_data.field_2) };
            let field3 = unsafe { NonNull::new_unchecked(&mut cas_data.field_3) };
            let guard = crossbeam_epoch::pin();
            let cas_change = CasChange::generate_for(&cas_data, &guard);

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
            mwcas.compare_exchange_u64(
                unsafe { &mut *field3.as_ptr() },
                cas_change.field_3_orig_val,
                cas_change.field_3_new_val,
            );
            mwcas.exec(&guard);
        })
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
