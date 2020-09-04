mod distributions;
use distributions::*;

fn main() {
    let n = Normal::new(2., 3.);
    let u = Uniform::new(1., 6.);
    let v: Vec<Box<dyn Distribution>> = vec![Box::new(n), Box::new(u)];
    for i in v {
        println!("{}", i.sample());
    }
}
