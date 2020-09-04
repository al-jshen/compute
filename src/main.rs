mod distributions;
mod functions;
use distributions::*;

fn main() {
    let n = Normal::new(2., 3.);
    let u = Uniform::new(1., 6.);
    let g = Gamma::new(2., 4.);
    let e = Exponential::new(3.);
    let b = Beta::new(2., 2.);
    let v: Vec<Box<dyn Distribution>> = vec![
        Box::new(n),
        Box::new(u),
        Box::new(g),
        Box::new(e),
        Box::new(b),
    ];
    for i in v {
        println!("{}", i.sample());
    }
}
