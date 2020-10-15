mod data;
mod distributions;
mod functions;
use data::*;
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
    let data = v.iter().map(|d| d.sample()).collect::<Vec<_>>();
    println!("{:?}", &data);
    println!("{}", variance(&data));
    let data1: Vec<f64> = vec![
        -0.2711336,
        1.20002575,
        0.69102151,
        -0.56390913,
        -1.62661382,
        -0.0613969,
        0.39876752,
        -0.99619281,
        1.12860854,
        -0.61163405,
    ];
}
