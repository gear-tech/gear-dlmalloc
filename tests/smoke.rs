extern crate dlmalloc;
extern crate rand;

use dlmalloc::Dlmalloc;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::cmp;

#[test]
fn smoke() {
    let mut a = Dlmalloc::new();
    unsafe {
        let ptr = a.malloc(1, 1);
        assert!(!ptr.is_null());
        *ptr = 9;
        assert_eq!(*ptr, 9);
        a.free(ptr, 1, 1);

        let ptr = a.malloc(1, 1);
        assert!(!ptr.is_null());
        *ptr = 10;
        assert_eq!(*ptr, 10);
        a.free(ptr, 1, 1);
    }
}

fn run_stress(seed: u64) {
    let mut a = Dlmalloc::new();

    println!("++++++++++++++++++++++ seed = {}\n", seed);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut ptrs = Vec::new();
    let max = if cfg!(test_lots) { 1_000_000 } else { 10_000 };
    unsafe {
        for _k in 0..max {
            let free = !ptrs.is_empty()
                && ((ptrs.len() < 10_000 && rng.gen_bool(1f64 / 3f64)) || rng.gen());
            if free {
                let idx = rng.gen_range(0, ptrs.len());
                let (ptr, size, align) = ptrs.swap_remove(idx);
                a.free(ptr, size, align);
                continue;
            }

            if !ptrs.is_empty() && rng.gen_bool(1f64 / 100f64) {
                let idx = rng.gen_range(0, ptrs.len());
                let (ptr, size, align) = ptrs.swap_remove(idx);
                let new_size = if rng.gen() {
                    rng.gen_range(size, size * 2)
                } else if size > 10 {
                    rng.gen_range(size / 2, size)
                } else {
                    continue;
                };
                let mut tmp = Vec::new();
                for i in 0..cmp::min(size, new_size) {
                    tmp.push(*ptr.add(i));
                }
                let ptr = a.realloc(ptr, size, align, new_size);
                assert!(!ptr.is_null());
                for (i, byte) in tmp.iter().enumerate() {
                    assert_eq!(*byte, *ptr.add(i));
                }
                ptrs.push((ptr, new_size, align));
            }

            let size = if rng.gen() {
                rng.gen_range(1, 128)
            } else {
                rng.gen_range(1, 128 * 1024)
            };
            let align = if rng.gen_bool(1f64 / 10f64) {
                1 << rng.gen_range(3, 8)
            } else {
                8
            };

            let zero = rng.gen_bool(1f64 / 50f64);
            let ptr = if zero {
                a.calloc(size, align)
            } else {
                a.malloc(size, align)
            };
            for i in 0..size {
                if zero {
                    assert_eq!(*ptr.add(i), 0);
                }
                *ptr.add(i) = 0xce;
            }
            ptrs.push((ptr, size, align));
        }
    }
}

#[test]
fn many_stress() {
    for i in 0..200 {
        run_stress(i);
    }
}

#[test]
fn stress() {
    let mut rng = rand::thread_rng();
    let seed: u64 = rng.gen();
    let seed = seed % 10000;
    run_stress(seed);
}
