use anyhow::Result;
use opencv::{self as cv};

use crate::Template;

#[derive(Debug, Clone)]
pub struct TemplateMatcher {
    descriptors: Vec<Template>,
}

impl TemplateMatcher {
    pub fn new(descriptors: &[Template]) -> Result<Self> {
        Ok(Self {
            descriptors: descriptors.to_vec(),
        })
    }

    pub fn descriptors(&self) -> &[Template] {
        &self.descriptors
    }

    pub fn find_descriptor(&self, label: &str) -> Option<&Template> {
        self.descriptors.iter().find(|d| d.label == label)
    }

    pub fn run_match(&self, input: &cv::core::Mat) -> Result<Vec<TemplateMatcherResult>> {
        let mut results = Vec::new();
        for descriptor in &self.descriptors {
            results.push(TemplateMatcherResult::new(
                descriptor.label.clone(),
                descriptor.run_match(input)?,
            ));
        }
        Ok(results)
    }
}

#[derive(Debug, Clone)]
pub struct TemplateMatcherResult {
    label: String,
    match_mat: cv::core::Mat,
}

impl TemplateMatcherResult {
    pub(crate) fn new(label: String, mat: cv::core::Mat) -> Self {
        Self {
            label,
            match_mat: mat,
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn mat(&self) -> &cv::core::Mat {
        &self.match_mat
    }

    pub fn min_max_loc(&self) -> Result<(f64, f64, cv::core::Point, cv::core::Point)> {
        let mut min_val = 0.0f64;
        let mut max_val = 0.0f64;
        let mut min_loc = cv::core::Point::default();
        let mut max_loc = cv::core::Point::default();
        cv::core::min_max_loc(
            &self.match_mat,
            Some(&mut min_val),
            Some(&mut max_val),
            Some(&mut min_loc),
            Some(&mut max_loc),
            &cv::core::no_array(),
        )?;
        Ok((min_val, max_val, min_loc, max_loc))
    }

    pub fn position(&self) -> Result<cv::core::Point> {
        let (_min_val, _max_val, _min_loc, max_loc) = self.min_max_loc()?;
        Ok(max_loc)
    }
}

impl From<TemplateMatcherResult> for cv::core::Mat {
    fn from(res: TemplateMatcherResult) -> Self {
        res.match_mat
    }
}

impl AsRef<cv::core::Mat> for TemplateMatcherResult {
    fn as_ref(&self) -> &cv::core::Mat {
        &self.match_mat
    }
}
