use crate::distributions::{Distribution, Normal};
use crate::summary::{max, min};
use std::f64::consts::PI;

pub fn sim_anneal<F>(
    f: F,
    lower_bound: f64,
    upper_bound: f64,
    nsteps: usize,
    init_temp: f64,
) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let prop_gen = Normal::new(0., 0.3);
    let mut current_state = (lower_bound + upper_bound) / 2.;
    for i in 0..nsteps {
        let temp = init_temp * ((PI * ((i + 1) as f64 / nsteps as f64)).cos() + 1.) / 2.;
        let prop_state = current_state + prop_gen.sample();
        let prop_state = max(&[min(&[prop_state, upper_bound].to_vec()), lower_bound].to_vec());
        let current_energy = f(current_state);
        let prop_energy = f(prop_state);
        let move_prob = if current_energy < prop_energy {
            1.
        } else {
            (-(current_energy - prop_energy) / temp).exp()
        };
        println!("{}", move_prob);
        if move_prob >= fastrand::f64() {
            current_state = prop_state;
        }
    }
    (current_state, f(current_state))
}
