use anyhow::Result;
use ndarray as nd;
use opencv as cv;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::nms::nms;

use super::{NmsConfig, TemplateMatch, TemplateModel};

#[derive(Debug, Clone)]
pub struct MultiMatcherDescriptor {
    pub label: String,
    pub matcher: TemplateModel,
}

impl MultiMatcherDescriptor {
    pub fn new(label: impl Into<String>, matcher: TemplateModel) -> Self {
        Self {
            label: label.into(),
            matcher,
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

    pub fn add_model(mut self, label: impl Into<String>, matcher: TemplateModel) -> Self {
        self.descriptors
            .push(MultiMatcherDescriptor::new(label, matcher));
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

    pub fn find_matches(&self, input: &cv::core::Mat) -> Result<Vec<MultiMatcherResult>> {
        let every_matches = self
            .descriptors
            .par_iter()
            .map(|d| {
                d.matcher
                    .match_all(input)
                    .map(|matches| {
                        matches
                            .into_iter()
                            .map(|result| MultiMatcherResult {
                                label: d.label.clone(),
                                result,
                            })
                            .collect::<Vec<_>>()
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(every_matches.into_iter().flatten().collect())
    }

    pub fn find_best_matches(
        &self,
        input: &cv::core::Mat,
        nms_config: NmsConfig,
    ) -> Result<Vec<MultiMatcherResult>> {
        let matches = self.find_matches(input)?;
        if matches.is_empty() {
            return Ok(matches);
        }

        let keep = calc_nms_indices(&matches, nms_config);
        let mut filtered = keep.iter().map(|&i| matches[i].clone()).collect::<Vec<_>>();
        filtered.sort_by(|a, b| {
            b.result
                .score
                .partial_cmp(&a.result.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(filtered)
    }
}

#[derive(Debug, Clone)]
pub struct MultiMatcherResult {
    pub label: String,
    pub result: TemplateMatch,
}

fn calc_nms_indices(results: &[MultiMatcherResult], config: NmsConfig) -> Vec<usize> {
    let (boxes, scores) = calc_nms_scores(results);
    nms(&boxes, &scores, config.iou_threshold, config.score_threshold)
}

fn calc_nms_scores(results: &[MultiMatcherResult]) -> (nd::Array2<i32>, nd::Array1<f64>) {
    let (boxes, scores) = results.iter().filter(|r| !r.result.size.empty()).fold(
        (nd::Array2::<i32>::default((0, 4)), Vec::new()),
        |(mut boxes, mut scores), result| {
            boxes
                .push(
                    nd::Axis(0),
                    nd::ArrayView::from(&[
                        result.result.position.x,
                        result.result.position.y,
                        result.result.position.x + result.result.size.width,
                        result.result.position.y + result.result.size.height,
                    ]),
                )
                .unwrap();
            scores.push(result.result.score as f64);
            (boxes, scores)
        },
    );
    (boxes, nd::Array1::from(scores))
}
