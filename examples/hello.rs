#![cfg(feature = "global")]

extern crate gear_dlmalloc;
extern crate rand;

use std::collections::LinkedList;

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

#[global_allocator]
static A: gear_dlmalloc::GlobalDlmalloc = gear_dlmalloc::GlobalDlmalloc;

#[derive(Debug, Clone, Copy)]
struct Point {
    x: u64,
    y: u64,
}

struct Rectangle {
    top_left: Point,
    bottom_right: Point,
}

#[inline(never)]
fn test1() {
    {
        let p1 = Box::new(Point { x: 0, y: 0 });
        let p2 = Box::new(Point { x: 1, y: 2 });
        let r = Box::new(Rectangle {
            top_left: *p1,
            bottom_right: *p2,
        });
        drop((p1, p2));
        assert!(r.top_left.x == 0);
        assert!(r.top_left.y == 0);
        assert!(r.bottom_right.x == 1);
        assert!(r.bottom_right.y == 2);
    }
}

#[inline(never)]
fn test2() {
    {
        let mut rng = rand::thread_rng();
        let seed: u64 = rng.gen();
        let seed = seed % 10000;
        println!("+++++++++++++++ seed == {}", seed);
        let mut rng = StdRng::seed_from_u64(seed);

        let mut v1: Vec<u64> = Vec::new();
        let mut v2: Vec<u64> = Vec::new();
        let mut l1: LinkedList<u8> = LinkedList::new();
        let mut l2: LinkedList<u8> = LinkedList::new();

        for i in 0..100 {
            println!("{}", i);
            let rem_number: usize = rng.gen_range(0..100);
            let add_number: usize = rng.gen_range(rem_number..rem_number + 10);
            for _ in 0..add_number {
                let val: u64 = rng.gen();
                v1.push(val);
                v2.push(val);
                let val: u8 = rng.gen();
                l1.push_back(val);
                l2.push_back(val);
            }
            for _ in 0..rem_number {
                assert_eq!(v1.len(), v2.len());
                let index: usize = rng.gen_range(0..v1.len());
                v1.remove(index);
                v2.remove(index);
                l1.pop_back();
                l2.pop_back();
            }
        }
        assert_eq!(v1, v2);
        assert_eq!(l1, l2);
    }
}

fn main() {
    test1();
    test2();
}
