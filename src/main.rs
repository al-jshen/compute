mod distributions;
use distributions::*;

fn main() {
    let n = distributions::normal::Normal::new(4., 5.);
    println!("d={:?}", &n.sample_iter(10));
}
