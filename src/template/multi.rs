use anyhow::Result;
use opencv::{self as cv};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::{FindBestMatchesConfig, MatchResult, Template};

#[derive(Debug, Clone)]
pub struct MultiMatcherDescriptor {
    pub label: String,
    pub template: Template,
}

impl MultiMatcherDescriptor {
    pub fn new(label: String, template: Template) -> Self {
        Self { label, template }
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
            .flat_map(|d| match d.template.find_matching_points(input) {
                Ok(matches) => matches
                    .into_iter()
                    .map(move |m| {
                        Ok(MultiMatcherResult {
                            label: d.label.clone(),
                            result: m,
                        })
                    })
                    .collect::<Vec<_>>(),
                Err(e) => vec![Err(e)],
            })
            .collect::<Result<Vec<_>, _>>()?;
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
pub struct MultiMatcherResult {
    pub label: String,
    pub result: MatchResult,
}
