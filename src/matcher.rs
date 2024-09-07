use anyhow::Result;
use opencv::{self as cv};

use crate::TemplateDescriptor;

#[derive(Debug, Clone)]
pub struct TemplateMatcher {
    descriptors: Vec<TemplateDescriptor>,
}

impl TemplateMatcher {
    pub fn new(descriptors: &[TemplateDescriptor]) -> Result<Self> {
        Ok(Self {
            descriptors: descriptors.to_vec(),
        })
    }

    pub fn descriptors(&self) -> &[TemplateDescriptor] {
        &self.descriptors
    }

    pub fn find_descriptor(&self, label: &str) -> Option<&TemplateDescriptor> {
        self.descriptors.iter().find(|d| d.label == label)
    }

    pub fn run_match(&self, input: &cv::core::Mat) -> Result<Vec<TemplateMatcherResult>> {
        let mut results = Vec::new();
        for descriptor in &self.descriptors {
            let mut res = cv::core::Mat::default();
            if let Some(mask) = &descriptor.mask {
                cv::imgproc::match_template(
                    &mat_to_grayscale(input)?,
                    &descriptor.template,
                    &mut res,
                    descriptor
                        .matching_method
                        .unwrap_or(cv::imgproc::TM_CCOEFF_NORMED),
                    mask,
                )?;
            } else {
                cv::imgproc::match_template(
                    &mat_to_grayscale(input)?,
                    &descriptor.template,
                    &mut res,
                    descriptor
                        .matching_method
                        .unwrap_or(cv::imgproc::TM_CCOEFF_NORMED),
                    &cv::core::no_array(),
                )?;
            }
            results.push(TemplateMatcherResult::new(descriptor.label.clone(), res));
        }
        Ok(results)
    }
}

fn mat_to_grayscale(mat: &cv::core::Mat) -> Result<cv::core::Mat> {
    let mut res = cv::core::Mat::default();
    cv::imgproc::cvt_color(&mat, &mut res, cv::imgproc::COLOR_RGBA2GRAY, 0)?;
    Ok(res)
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
