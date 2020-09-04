mod distributions;
use distributions::*;

fn main() {
    let n = Normal::new(2., 3.);
    let u = Uniform::new(1., 6.);
    let g = Gamma::new(2., 4.);
    let v: Vec<Box<dyn Distribution>> = vec![Box::new(n), Box::new(u), Box::new(g)];
    println!("d={:?}", v[2].sample_iter(1000));
    // for i in v {
    //     println!("{}", i.sample());
    // }
}
