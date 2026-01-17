use anyhow::Result;
use opencv as cv;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::{FindBestMatchesConfig, MatchResult, Template};

#[derive(Debug, Clone)]
pub struct TemplatePyramidLevel {
    pub scale: f64,
    pub template: Template,
}

#[derive(Debug, Clone)]
pub struct TemplatePyramid {
    levels: Vec<TemplatePyramidLevel>,
}

impl TemplatePyramid {
    pub fn new(base: Template, mut scales: Vec<f64>) -> Result<Self> {
        if scales.is_empty() {
            scales.push(1.0);
        }
        let mut levels = Vec::with_capacity(scales.len());
        for scale in scales {
            let template = base.resized_with_scale(scale)?;
            levels.push(TemplatePyramidLevel { scale, template });
        }
        Ok(Self { levels })
    }

    pub fn levels(&self) -> &[TemplatePyramidLevel] {
        &self.levels
    }

    pub fn find_matching_points(
        &self,
        input: &cv::core::Mat,
    ) -> Result<Vec<TemplatePyramidMatchResult>> {
        let every_matches = self
            .levels
            .par_iter()
            .flat_map(|level| match level.template.find_matching_points(input) {
                Ok(matches) => matches
                    .into_iter()
                    .map(move |m| {
                        Ok(TemplatePyramidMatchResult {
                            scale: level.scale,
                            result: m,
                        })
                    })
                    .collect::<Vec<_>>(),
                Err(e) => vec![Err(e)],
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(every_matches)
    }

    pub fn find_best_matches(
        &self,
        input: &cv::core::Mat,
        config: FindBestMatchesConfig,
    ) -> Result<Vec<TemplatePyramidMatchResult>> {
        let every_matches = self.find_matching_points(input)?;
        let keep = MatchResult::calc_nms_indices(
            &every_matches
                .iter()
                .map(|m| m.result.clone())
                .collect::<Vec<_>>(),
            config.iou_threshold,
            config.score_threshold,
        );
        Ok(keep.iter().map(|&i| every_matches[i].clone()).collect())
    }
}

impl Template {
    pub fn pyramid(&self, scales: Vec<f64>) -> Result<TemplatePyramid> {
        TemplatePyramid::new(self.clone(), scales)
    }
}

#[derive(Debug, Clone)]
pub struct TemplatePyramidMatchResult {
    pub scale: f64,
    pub result: MatchResult,
}
