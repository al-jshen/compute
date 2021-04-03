use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Formula<'a, 'b> {
    formula: &'a str,
    data: HashMap<&'b str, Vec<f64>>,
}

impl<'a, 'b> Formula<'a, 'b> {
    pub fn new(formula: &'a str, data: HashMap<&'b str, Vec<f64>>) -> Self {
        Self { formula, data }
    }

    pub fn parse(&self) -> Result<Vec<f64>, &'static str> {
        assert!(self.formula.contains("~"), "Formula must contain ~.");
        todo!()
    }
}
