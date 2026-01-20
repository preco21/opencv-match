use anyhow::Result;
use ndarray as nd;
use opencv as cv;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::nms::nms;

use super::core::internal::pose_to_box;
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
                acc.extend(d.matcher.find_matching_points(input, &d.label, &match_config)?);
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

        nms_filter_results(&results, nms_config)
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

fn nms_filter_results(
    results: &[MultiMatcherResult],
    nms_config: NmsConfig,
) -> Result<Vec<MultiMatcherResult>> {
    if results.is_empty() {
        return Ok(Vec::new());
    }

    let (boxes, scores) = nms_inputs_from_results(results)?;
    if boxes.nrows() == 0 {
        return Ok(Vec::new());
    }

    let keep = nms(
        &boxes,
        &scores,
        nms_config.iou_threshold,
        nms_config.score_threshold,
    );
    Ok(keep.iter().map(|&idx| results[idx].clone()).collect())
}

fn nms_inputs_from_results(
    results: &[MultiMatcherResult],
) -> Result<(nd::Array2<i32>, nd::Array1<f64>)> {
    let mut boxes = nd::Array2::<i32>::default((0, 4));
    let mut scores = Vec::with_capacity(results.len());
    for result in results {
        let rect = pose_to_box(&result.pose)?;
        boxes.push(nd::Axis(0), nd::ArrayView::from(&rect)).unwrap();
        scores.push(result.pose.score as f64);
    }
    Ok((boxes, nd::Array1::from(scores)))
}
