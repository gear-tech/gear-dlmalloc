#![cfg(feature = "global")]

extern crate gear_dlmalloc;

use std::collections::HashMap;
use std::thread;

#[global_allocator]
static A: gear_dlmalloc::GlobalDlmalloc = gear_dlmalloc::GlobalDlmalloc;

#[test]
fn foo() {
    println!("hello");
}

#[test]
fn map() {
    let mut m = HashMap::new();
    m.insert(1, 2);
    m.insert(5, 3);
    drop(m);
}

#[test]
fn strings() {
    format!("foo, bar, {}", "baz");
}

#[test]
fn threads() {
    assert!(thread::spawn(|| panic!()).join().is_err());
}
