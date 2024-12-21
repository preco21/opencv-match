use ndarray as nd;
use opencv::{self as cv, core::MatTraitConst};

use crate::convert;

#[derive(Debug, Clone)]
pub struct Template {
    threshold: f32,
    matching_method: Option<i32>,
    template: cv::core::Mat,
    mask: Option<cv::core::Mat>,
    source_template: cv::core::Mat,
    source_mask: Option<cv::core::Mat>,
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

#[derive(Debug)]
pub struct FindBestMatchesConfig {
    /// The threshold for the Intersection over Union (IoU) to use for non-maximum suppression.
    /// If not provided, `0.1` will be used.
    pub iou_threshold: f64,
    /// The threshold for the score to use for non-maximum suppression.
    /// If not provided, `0.1` will be used.
    pub score_threshold: f64,
}

impl Default for FindBestMatchesConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.1,
            score_threshold: 0.1,
        }
    }
}

impl Template {
    pub fn new(template: cv::core::Mat, config: TemplateConfig) -> anyhow::Result<Self> {
        Ok(Self {
            threshold: config.threshold,
            matching_method: config.matching_method,
            template: template.clone(),
            source_template: template,
            mask: None,
            source_mask: None,
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
            source_template: template,
            mask: Some(mask.clone()),
            source_mask: Some(mask),
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

    pub fn mask(&self) -> Option<&cv::core::Mat> {
        self.mask.as_ref()
    }

    pub fn width(&self) -> i32 {
        self.template.cols()
    }

    pub fn height(&self) -> i32 {
        self.template.rows()
    }

    pub fn resize(&mut self, width: i32, height: i32) -> anyhow::Result<()> {
        let mut res = cv::core::Mat::default();
        cv::imgproc::resize(
            &self.source_template,
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
        if let Some(mask) = &self.source_mask {
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

    pub fn resize_with_scale(&mut self, scale: f64) -> anyhow::Result<()> {
        let mut res = cv::core::Mat::default();
        cv::imgproc::resize(
            &self.source_template,
            &mut res,
            Default::default(),
            scale,
            scale,
            cv::imgproc::INTER_NEAREST_EXACT,
        )?;
        self.source_template = res;
        Ok(())
    }

    pub fn resize_mask_with_scale(&mut self, scale: f64) -> anyhow::Result<()> {
        if let Some(mask) = &self.source_mask {
            let mut res = cv::core::Mat::default();
            cv::imgproc::resize(
                &mask,
                &mut res,
                Default::default(),
                scale,
                scale,
                cv::imgproc::INTER_NEAREST_EXACT,
            )?;
            self.source_mask = Some(res);
        }
        Ok(())
    }

    pub fn find_best_matches(
        &self,
        input: &cv::core::Mat,
        config: FindBestMatchesConfig,
    ) -> anyhow::Result<Vec<MatchResult>> {
        let res = self.find_all_matches(input)?;
        let (boxes, scores) = res.iter().fold(
            (nd::Array2::<f64>::default((res.len(), 4)), Vec::new()),
            |(mut boxes, mut scores), result| {
                boxes
                    .push(
                        nd::Axis(0),
                        nd::ArrayView::from(&[
                            result.position.0 as f64,
                            result.dimension.0 as f64,
                            result.position.1 as f64,
                            result.dimension.1 as f64,
                        ]),
                    )
                    .unwrap();
                scores.push(result.score as f64);
                (boxes, scores)
            },
        );
        let keep = powerboxesrs::nms::rtree_nms(
            &boxes,
            &scores,
            config.iou_threshold,
            config.score_threshold,
        );
        Ok(keep.iter().map(|&i| res[i].clone()).collect())
    }

    pub fn find_all_matches(&self, input: &cv::core::Mat) -> anyhow::Result<Vec<MatchResult>> {
        let dimension = (self.width() as usize, self.height() as usize);
        let res = self.run_match(input)?;
        let buf = convert::mat_to_array2(&res)?;
        let indices: Vec<MatchResult> = nd::Zip::indexed(&buf).par_fold(
            || Vec::new(),
            |mut indices, (i, j), &val| {
                if val > self.threshold {
                    indices.push(MatchResult {
                        position: (i, j),
                        dimension,
                        score: val,
                    });
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
}

#[derive(Debug, Clone)]
pub struct MatchResult {
    pub position: (usize, usize),
    pub dimension: (usize, usize),
    pub score: f32,
}
