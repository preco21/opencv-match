use ndarray as nd;
use opencv as cv;

use crate::convert::{mat_to_array2, mat_to_grayscale};

#[derive(Debug, Clone)]
pub struct Template {
    pub threshold: f32,
    pub matching_method: Option<i32>,
    template: cv::core::Mat,
    original_template: cv::core::Mat,
    mask: Option<cv::core::Mat>,
    original_mask: Option<cv::core::Mat>,
}

impl Template {
    pub fn new(template: cv::core::Mat, threshold: f32) -> Self {
        Self {
            template: template.clone(),
            mask: None,
            threshold,
            matching_method: None,
            original_template: template,
            original_mask: None,
        }
    }

    pub fn with_mask(template: cv::core::Mat, mask: cv::core::Mat, threshold: f32) -> Self {
        Self {
            template: template.clone(),
            mask: Some(mask.clone()),
            threshold,
            matching_method: None,
            original_template: template,
            original_mask: Some(mask),
        }
    }

    pub fn resize(&mut self, width: i32, height: i32) -> anyhow::Result<()> {
        let mut res = cv::core::Mat::default();
        cv::imgproc::resize(
            &self.original_template,
            &mut res,
            cv::core::Size::new(width, height),
            0.0,
            0.0,
            // After some testing, I found that `INTER_NEAREST_EXACT` is the
            // sweet spot for resizing template images. It produces better
            // results than others in terms of maintaining pixel patterns as the original
            // template image provided.
            //
            // For more details:
            // https://stackoverflow.com/questions/5358700/template-match-different-sizes-of-template-and-image
            // See also:
            // https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121aa5521d8e080972c762467c45f3b70e6c
            cv::imgproc::INTER_NEAREST_EXACT,
        )?;
        self.template = res;
        Ok(())
    }

    pub fn resize_mask(&mut self, width: i32, height: i32) -> anyhow::Result<()> {
        if let Some(mask) = &self.original_mask {
            let mut res = cv::core::Mat::default();
            cv::imgproc::resize(
                &mask,
                &mut res,
                cv::core::Size::new(width, height),
                0.0,
                0.0,
                cv::imgproc::INTER_NEAREST_EXACT,
            )?;
            self.mask = Some(res);
        }
        Ok(())
    }

    pub fn resize_scale(&mut self, scale: f64) -> anyhow::Result<()> {
        let mut res = cv::core::Mat::default();
        cv::imgproc::resize(
            &self.original_template,
            &mut res,
            Default::default(),
            scale,
            scale,
            cv::imgproc::INTER_NEAREST_EXACT,
        )?;
        self.original_template = res;
        Ok(())
    }

    pub fn resize_scale_mask(&mut self, scale: f64) -> anyhow::Result<()> {
        if let Some(mask) = &self.original_mask {
            let mut res = cv::core::Mat::default();
            cv::imgproc::resize(
                &mask,
                &mut res,
                Default::default(),
                scale,
                scale,
                cv::imgproc::INTER_NEAREST_EXACT,
            )?;
            self.original_mask = Some(res);
        }
        Ok(())
    }

    pub fn run_match(&self, input: &cv::core::Mat) -> anyhow::Result<cv::core::Mat> {
        let mut res = cv::core::Mat::default();
        if let Some(mask) = &self.mask {
            cv::imgproc::match_template(
                &mat_to_grayscale(input)?,
                &self.template,
                &mut res,
                self.matching_method
                    .unwrap_or(cv::imgproc::TM_CCOEFF_NORMED),
                mask,
            )?;
        } else {
            cv::imgproc::match_template(
                &mat_to_grayscale(input)?,
                &self.template,
                &mut res,
                self.matching_method
                    .unwrap_or(cv::imgproc::TM_CCOEFF_NORMED),
                &cv::core::no_array(),
            )?;
        }
        Ok(res)
    }

    pub fn find_best_matches(&self, input: &cv::core::Mat) -> anyhow::Result<Vec<(usize, usize)>> {
        let res = self.run_match(input)?;
        let buf = mat_to_array2(&res)?;
        let indices: Vec<(usize, usize)> = nd::Zip::indexed(&buf).par_fold(
            || Vec::new(),
            |mut indices, (i, j), &val| {
                if val > self.threshold {
                    indices.push((i, j));
                }
                indices
            },
            |mut a, b| {
                a.extend(b);
                a
            },
        );
        Ok(indices)
    }
}
