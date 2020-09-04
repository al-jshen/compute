mod distributions;
mod functions;
use distributions::*;

fn main() {
    let mut n = Normal::default();
    println!("{:?}", n);
    n.set_mu(3.).set_sigma(7.);
    println!("{:?}", n);
    n.update(&[2., 4.]);
    println!("{:?}", n);
    let u = Uniform::new(1., 6.);
    let g = Gamma::new(2., 4.);
    let e = Exponential::new(3.);
    let b = Beta::new(2., 2.);
    let c = ChiSquared::new(6);
    let v: Vec<Box<dyn Distribution>> = vec![
        Box::new(n),
        Box::new(u),
        Box::new(g),
        Box::new(e),
        Box::new(b),
        Box::new(c),
    ];
    for i in v {
        println!("{}", i.sample());
    }
}
