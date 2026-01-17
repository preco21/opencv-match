use anyhow::Result;
use opencv::{self as cv};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::{FindBestMatchesConfig, MatchResult, Template, TemplatePyramid};

#[derive(Debug, Clone)]
pub struct MultiMatcherDescriptor {
    pub label: String,
    pub matcher: TemplateMatcher,
}

impl MultiMatcherDescriptor {
    pub fn new(label: String, template: Template) -> Self {
        Self {
            label,
            matcher: TemplateMatcher::Single(template),
        }
    }

    pub fn with_pyramid(label: String, pyramid: TemplatePyramid) -> Self {
        Self {
            label,
            matcher: TemplateMatcher::Pyramid(pyramid),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultiMatcher {
    descriptors: Vec<MultiMatcherDescriptor>,
}

/// A helper struct to hold multiple templates and run matching on all of them at once.
///
/// This also helps to find the best matches across all templates simultaneously.
impl MultiMatcher {
    pub fn new(descriptors: Vec<MultiMatcherDescriptor>) -> Result<Self> {
        Ok(Self { descriptors })
    }

    pub fn descriptors(&self) -> &[MultiMatcherDescriptor] {
        &self.descriptors
    }

    pub fn find_descriptor(&self, label: &str) -> Option<&MultiMatcherDescriptor> {
        self.descriptors.iter().find(|d| d.label == label)
    }

    pub fn find_best_matches(
        &self,
        input: &cv::core::Mat,
        config: FindBestMatchesConfig,
    ) -> Result<Vec<MultiMatcherResult>> {
        let every_matches = self
            .descriptors
            .par_iter()
            .map(|d| d.matcher.find_matching_points(input, &d.label))
            .collect::<Result<Vec<_>, _>>()?;
        let every_matches = every_matches.into_iter().flatten().collect::<Vec<_>>();
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

#[derive(Debug, Clone)]
pub enum TemplateMatcher {
    Single(Template),
    Pyramid(TemplatePyramid),
}

impl TemplateMatcher {
    fn find_matching_points(
        &self,
        input: &cv::core::Mat,
        label: &str,
    ) -> Result<Vec<MultiMatcherResult>> {
        match self {
            TemplateMatcher::Single(template) => Ok(template
                .find_matching_points(input)?
                .into_iter()
                .map(|result| MultiMatcherResult {
                    label: label.to_string(),
                    scale: None,
                    result,
                })
                .collect()),
            TemplateMatcher::Pyramid(pyramid) => Ok(pyramid
                .find_matching_points(input)?
                .into_iter()
                .map(|m| MultiMatcherResult {
                    label: label.to_string(),
                    scale: Some(m.scale),
                    result: m.result,
                })
                .collect()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultiMatcherResult {
    pub label: String,
    pub scale: Option<f64>,
    pub result: MatchResult,
}
