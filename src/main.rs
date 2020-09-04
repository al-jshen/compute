mod distributions;
use distributions::*;

fn main() {
    let n = distributions::normal::Normal::new(4., 5.);
    // println!("d={:?}", &n.sample_iter(1000));
    (-200..200).for_each(|x| println!("{}", &n.pdf(x as f64)));
}
