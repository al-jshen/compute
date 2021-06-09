use compute::prelude::*;
use plotly::{
    common::{Font, Line, Marker, Mode, Title},
    layout::{Axis, Legend},
    Layout, Plot, Rgb, Rgba, Scatter,
};

fn main() {
    // variance parameter
    let var = 0.05;
    // length-scale squared parameter
    let lsq = 1.2;
    // generate test points
    let n = 250;
    let xtest = linspace(0., 10., n).to_matrix().reshape(-1, 1);
    // similarity of test values
    let k_ss = rbfkernel(&xtest, &xtest, var, lsq);

    // 10 randomly sampled noiseless training points
    let xtrain = Uniform::new(0., 10.).sample_matrix(10, 1);
    let ytrain = xtrain.sin();

    // apply kernel to training points
    let kern = rbfkernel(&xtrain, &xtrain, var, lsq);
    let l = (&kern + Matrix::eye(xtrain.nrows) * 0.00005).cholesky();
    let (lu, piv) = l.lu();

    // get mean at test points
    let k_s = rbfkernel(&xtrain, &xtest, var, lsq);
    let lk = lu.lu_solve(&piv, &k_s);
    let mu = lk.t().dot(lu.lu_solve(&piv, &ytrain)).to_vec();

    // get uncertainty on prediction
    let std = (k_ss.diag() - lk.powi(2).sum_cols()).sqrt();

    // plot
    let layout = Layout::new()
        .title(
            Title::new(r"$\text{Gaussian Process Regression on }f(x) = \sin(x)$".into())
                .font(Font::new().size(30)),
        )
        .y_axis(Axis::new().title(Title::new("y")))
        .x_axis(Axis::new().title(Title::new("x")))
        .font(Font::new().size(20))
        .legend(Legend::new().font(Font::new().size(20)));

    let trace1 = Scatter::new(xtrain.to_vec(), ytrain.to_vec())
        .name("Training data")
        .mode(Mode::Markers)
        .marker(Marker::new().size(15));
    let trace2 = Scatter::new(xtest.to_vec(), xtest.sin().to_vec())
        .name("True function")
        .mode(Mode::Lines)
        .line(Line::new().dash(plotly::common::DashType::Dot));
    let trace3 = Scatter::new(xtest.to_vec(), mu.clone())
        .name("Mean prediction")
        .mode(Mode::Lines)
        .line(Line::new().color(Rgb::new(50, 120, 200)).width(3.));
    let trace4 = Scatter::new(xtest.to_vec(), mu.clone() - 2. * std.clone())
        .mode(Mode::Lines)
        .show_legend(false)
        .line(Line::new().color(Rgb::new(20, 20, 20)));
    let trace5 = Scatter::new(xtest.to_vec(), mu.clone() + 2. * std.clone())
        .mode(Mode::Lines)
        .line(Line::new().color(Rgb::new(20, 20, 20)))
        .name("95% confidence interval")
        .fill(plotly::common::Fill::ToNextY)
        .fill_color(Rgba::new(30, 30, 30, 0.2));

    let mut plot = Plot::new();
    plot.set_layout(layout);
    plot.add_trace(trace4);
    plot.add_trace(trace5);
    plot.add_trace(trace2);
    plot.add_trace(trace1);
    plot.add_trace(trace3);
    // plot.show_png(1920, 1080);
    plot.show();
}

fn rbfkernel(a: &Matrix, b: &Matrix, var: f64, lengthsq: f64) -> Matrix {
    let sq_dist = a.powi(2).reshape(-1, 1) + b.powi(2).reshape(1, -1) - 2. * a.dot_t(b);
    var * (-0.5 * sq_dist / lengthsq).exp()
}
