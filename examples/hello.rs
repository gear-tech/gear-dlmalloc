extern crate dlmalloc;
extern crate rand;

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

#[global_allocator]
static A: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

struct Rectangle {
    top_left: Point,
    bottom_right: Point,
}

#[cfg(unix)]
#[cfg(target_pointer_width = "64")]
const END_ALLOCED_SIZE: usize = 0x60;

#[inline(never)]
fn test1() {
    {
        let p1 = Box::new(Point { x: 0f64, y: 0f64 });
        let p2 = Box::new(Point { x: 1f64, y: 2f64 });
        let r = Box::new(Rectangle {
            top_left: *p1,
            bottom_right: *p2,
        });
        drop((p1, p2));
        assert!(r.top_left.x == 0f64);
        assert!(r.top_left.y == 0f64);
        assert!(r.bottom_right.x == 1f64);
        assert!(r.bottom_right.y == 2f64);
    }
    let x: usize;
    unsafe {
        x = dlmalloc::get_alloced_mem_size();
    }
    assert_eq!(x, END_ALLOCED_SIZE);
}

#[inline(never)]
fn test2() {
    {
        let mut rng = rand::thread_rng();
        let seed: u64 = rng.gen();
        let seed = seed % 10000;
        let mut rng = StdRng::seed_from_u64(seed);

        let mut v1: Vec<u64> = Vec::new();
        let mut v2: Vec<u64> = Vec::new();

        for _ in 0..1000 {
            let rem_number: usize = rng.gen_range(0, 1000);
            let add_number: usize = rng.gen_range(rem_number, rem_number + 100);
            for _ in 0..add_number {
                let val: u64 = rng.gen();
                v1.push(val);
                v2.push(val);
            }
            for _ in 0..rem_number {
                assert_eq!(v1.len(), v2.len());
                let index: usize = rng.gen_range(0, v1.len());
                v1.remove(index);
                v2.remove(index);
            }
            assert_eq!(v1, v2);
        }
    }

    let x: usize;
    unsafe {
        x = dlmalloc::get_alloced_mem_size();
    }
    assert_eq!(x, END_ALLOCED_SIZE);
}

fn main() {
    test1();
    test2();
}
