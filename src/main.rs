mod distributions;
use distributions::*;

fn main() {
    let n = Normal::new(2., 3.);
    let u = Uniform::new(1., 6.);
    let g = Gamma::new(2., 4.);
    let e = Exponential::new(3.);
    let v: Vec<Box<dyn Distribution>> = vec![Box::new(n), Box::new(u), Box::new(g), Box::new(e)];
    // println!("d={:?}", v[2].sample_iter(1000));
    for i in v {
        println!("{}", i.sample());
    }
}
