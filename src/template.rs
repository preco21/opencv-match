use ndarray as nd;
use opencv as cv;

use crate::convert;

#[derive(Debug, Clone)]
pub struct Template {
    threshold: f32,
    matching_method: Option<i32>,
    template: cv::core::Mat,
    original_template: cv::core::Mat,
    mask: Option<cv::core::Mat>,
    original_mask: Option<cv::core::Mat>,
}

#[derive(Debug)]
pub struct TemplateConfig {
    /// The threshold to use for matching. If not provided, `0.8` will be used.
    pub threshold: f32,
    /// The method to use for matching. If not provided, `TM_CCOEFF_NORMED` will be used.
    pub matching_method: Option<i32>,
}

impl Default for TemplateConfig {
    fn default() -> Self {
        Self {
            threshold: 0.8,
            matching_method: None,
        }
    }
}

impl Template {
    pub fn new(template: cv::core::Mat, config: TemplateConfig) -> anyhow::Result<Self> {
        Ok(Self {
            threshold: config.threshold,
            matching_method: config.matching_method,
            template: template.clone(),
            original_template: template,
            mask: None,
            original_mask: None,
        })
    }

    pub fn with_mask(
        template: cv::core::Mat,
        mask: cv::core::Mat,
        config: TemplateConfig,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            threshold: config.threshold,
            matching_method: config.matching_method,
            template: template.clone(),
            original_template: template,
            mask: Some(mask.clone()),
            original_mask: Some(mask),
        })
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn matching_method(&self) -> Option<i32> {
        self.matching_method
    }

    pub fn template(&self) -> &cv::core::Mat {
        &self.template
    }

    pub fn original_template(&self) -> &cv::core::Mat {
        &self.original_template
    }

    pub fn mask(&self) -> Option<&cv::core::Mat> {
        self.mask.as_ref()
    }

    pub fn original_mask(&self) -> Option<&cv::core::Mat> {
        self.original_mask.as_ref()
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
                input,
                &self.template,
                &mut res,
                self.matching_method
                    .unwrap_or(cv::imgproc::TM_CCOEFF_NORMED),
                mask,
            )?;
        } else {
            cv::imgproc::match_template(
                input,
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
        let buf = convert::mat_to_array2(&res)?;
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
