use anyhow::Result;
use opencv as cv;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::core::{MatchConfig, NmsConfig, Pose};
use super::{Model, ModelPyramid};

#[derive(Debug, Clone)]
pub struct MultiMatcherDescriptor {
    pub label: String,
    pub matcher: ModelMatcher,
}

impl MultiMatcherDescriptor {
    pub fn new(label: String, model: Model) -> Self {
        Self {
            label,
            matcher: ModelMatcher::Single(model),
        }
    }

    pub fn with_pyramid(label: String, pyramid: ModelPyramid) -> Self {
        Self {
            label,
            matcher: ModelMatcher::Pyramid(pyramid),
        }
    }

    pub fn builder() -> MultiMatcherBuilder {
        MultiMatcherBuilder::new()
    }
}

#[derive(Debug, Clone)]
pub struct MultiMatcherBuilder {
    descriptors: Vec<MultiMatcherDescriptor>,
}

impl MultiMatcherBuilder {
    pub fn new() -> Self {
        Self {
            descriptors: Vec::new(),
        }
    }

    pub fn add_model(mut self, label: impl Into<String>, model: Model) -> Self {
        self.descriptors
            .push(MultiMatcherDescriptor::new(label.into(), model));
        self
    }

    pub fn add_pyramid(mut self, label: impl Into<String>, pyramid: ModelPyramid) -> Self {
        self.descriptors
            .push(MultiMatcherDescriptor::with_pyramid(label.into(), pyramid));
        self
    }

    pub fn add_descriptor(mut self, descriptor: MultiMatcherDescriptor) -> Self {
        self.descriptors.push(descriptor);
        self
    }

    pub fn build(self) -> MultiMatcher {
        MultiMatcher::new(self.descriptors)
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
    pub fn new(descriptors: Vec<MultiMatcherDescriptor>) -> Self {
        Self { descriptors }
    }

    pub fn descriptors(&self) -> &[MultiMatcherDescriptor] {
        &self.descriptors
    }

    pub fn find_descriptor(&self, label: &str) -> Option<&MultiMatcherDescriptor> {
        self.descriptors.iter().find(|d| d.label == label)
    }

    pub fn find_matching_points(
        &self,
        input: &cv::core::Mat,
        match_config: MatchConfig,
    ) -> Result<Vec<MultiMatcherResult>> {
        self.descriptors
            .par_iter()
            .try_fold(Vec::new, |mut acc, d| -> Result<Vec<MultiMatcherResult>> {
                acc.extend(
                    d.matcher
                        .find_matching_points(input, &d.label, &match_config)?,
                );
                Ok(acc)
            })
            .try_reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                Ok(a)
            })
    }

    pub fn find_best_matches(
        &self,
        input: &cv::core::Mat,
        match_config: MatchConfig,
        nms_config: NmsConfig,
    ) -> Result<Vec<MultiMatcherResult>> {
        let results = self.find_matching_points(input, match_config)?;
        if results.is_empty() {
            return Ok(results);
        }

        MultiMatcherResult::nms_filter(&results, nms_config)
    }
}

#[derive(Debug, Clone)]
pub enum ModelMatcher {
    Single(Model),
    Pyramid(ModelPyramid),
}

impl ModelMatcher {
    fn find_matching_points(
        &self,
        input: &cv::core::Mat,
        label: &str,
        match_config: &MatchConfig,
    ) -> Result<Vec<MultiMatcherResult>> {
        match self {
            ModelMatcher::Single(model) => Ok(model
                .match_model(input, match_config.clone())?
                .into_iter()
                .map(|pose| MultiMatcherResult {
                    label: label.to_string(),
                    scale: None,
                    pose,
                })
                .collect()),
            ModelMatcher::Pyramid(pyramid) => Ok(pyramid
                .find_matching_points(input, match_config.clone())?
                .into_iter()
                .map(|matched| MultiMatcherResult {
                    label: label.to_string(),
                    scale: Some(matched.scale),
                    pose: matched.pose,
                })
                .collect()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultiMatcherResult {
    pub label: String,
    pub scale: Option<f64>,
    pub pose: Pose,
}

impl MultiMatcherResult {
    pub fn nms_filter(
        results: &[MultiMatcherResult],
        nms_config: NmsConfig,
    ) -> Result<Vec<MultiMatcherResult>> {
        if results.is_empty() {
            return Ok(Vec::new());
        }

        let poses: Vec<Pose> = results.iter().map(|result| result.pose).collect();
        let keep = Pose::nms_filter_indices(&poses, nms_config)?;
        Ok(keep.iter().map(|&idx| results[idx].clone()).collect())
    }
}
