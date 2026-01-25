use anyhow::{bail, ensure, Result};
use ndarray as nd;
use opencv::imgproc;
use opencv::{self as cv, core::MatTraitConst, core::CV_32F, prelude::*};
use rayon::prelude::*;

use crate::nms::nms;

const TOLERANCE: f64 = 1.0e-7;

#[derive(Debug, Clone, Copy)]
pub struct NmsConfig {
    /// IoU threshold for NMS suppression.
    pub iou_threshold: f64,
    /// Score threshold for NMS pre-filtering.
    pub score_threshold: f64,
}

impl Default for NmsConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.3,
            score_threshold: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScaleConfig {
    pub min: f64,
    pub max: f64,
    pub steps: usize,
    pub scales: Option<Vec<f64>>,
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self {
            min: 0.5,
            max: 2.0,
            steps: 60,
            scales: None,
        }
    }
}

impl ScaleConfig {
    pub fn range(min: f64, max: f64, steps: usize) -> Self {
        Self {
            min,
            max,
            steps,
            scales: None,
        }
    }

    pub fn explicit(scales: Vec<f64>) -> Self {
        Self {
            min: 1.0,
            max: 1.0,
            steps: scales.len().max(1),
            scales: Some(scales),
        }
    }

    fn build_scales(&self) -> Result<Vec<f64>> {
        let mut scales = if let Some(scales) = &self.scales {
            scales.clone()
        } else {
            ensure!(self.steps > 0, "scale steps must be > 0");
            ensure!(
                self.min > 0.0 && self.max > 0.0,
                "scale range must be positive"
            );

            if (self.max - self.min).abs() <= TOLERANCE {
                vec![self.min]
            } else if self.steps == 1 {
                vec![(self.min + self.max) * 0.5]
            } else {
                let step = (self.max - self.min) / (self.steps as f64 - 1.0);
                (0..self.steps)
                    .map(|i| self.min + step * i as f64)
                    .collect()
            }
        };

        scales.retain(|s| *s > 0.0 && s.is_finite());
        scales.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        scales.dedup_by(|a, b| (*a - *b).abs() <= TOLERANCE);
        ensure!(!scales.is_empty(), "scales must not be empty");
        Ok(scales)
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    pub scales: Vec<f64>,
    pub fallback_ratio: f32,
    pub fallback_min: f32,
    pub fallback_max: f32,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            scales: vec![0.9, 0.95, 1.0, 1.05, 1.1],
            fallback_ratio: 0.5,
            fallback_min: 0.2,
            fallback_max: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TemplateModelConfig {
    pub threshold: f32,
    pub matching_method: i32,
    pub scale: ScaleConfig,
    pub nms: NmsConfig,
    pub max_matches: Option<usize>,
    pub per_scale_max: Option<usize>,
    pub adaptive: Option<AdaptiveConfig>,
}

impl Default for TemplateModelConfig {
    fn default() -> Self {
        Self {
            threshold: 0.7,
            matching_method: imgproc::TM_CCOEFF_NORMED,
            scale: ScaleConfig::default(),
            nms: NmsConfig::default(),
            max_matches: None,
            per_scale_max: None,
            adaptive: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TemplateMatch {
    pub position: cv::core::Point,
    pub size: cv::core::Size,
    pub score: f32,
    pub scale: f64,
}

impl TemplateMatch {
    pub fn calc_nms_indices(results: &[TemplateMatch], config: NmsConfig) -> Vec<usize> {
        let (boxes, scores) = Self::calc_nms_scores(results);
        nms(
            &boxes,
            &scores,
            config.iou_threshold,
            config.score_threshold,
        )
    }

    pub fn calc_nms_scores(results: &[TemplateMatch]) -> (nd::Array2<i32>, nd::Array1<f64>) {
        let (boxes, scores) = results.iter().filter(|r| !r.size.empty()).fold(
            (nd::Array2::<i32>::default((0, 4)), Vec::new()),
            |(mut boxes, mut scores), result| {
                boxes
                    .push(
                        nd::Axis(0),
                        nd::ArrayView::from(&[
                            result.position.x,
                            result.position.y,
                            result.position.x + result.size.width,
                            result.position.y + result.size.height,
                        ]),
                    )
                    .unwrap();
                scores.push(result.score as f64);
                (boxes, scores)
            },
        );
        (boxes, nd::Array1::from(scores))
    }
}

#[derive(Debug, Clone)]
pub struct TemplateModelBuilder {
    template: cv::core::Mat,
    mask: Option<cv::core::Mat>,
    config: TemplateModelConfig,
}

impl TemplateModelBuilder {
    pub fn new(template: cv::core::Mat) -> Self {
        Self {
            template,
            mask: None,
            config: TemplateModelConfig::default(),
        }
    }

    pub fn with_mask(mut self, mask: cv::core::Mat) -> Self {
        self.mask = Some(mask);
        self
    }

    pub fn threshold(mut self, threshold: f32) -> Self {
        self.config.threshold = threshold;
        self
    }

    pub fn matching_method(mut self, method: i32) -> Self {
        self.config.matching_method = method;
        self
    }

    pub fn scale_range(mut self, min: f64, max: f64, steps: usize) -> Self {
        self.config.scale = ScaleConfig::range(min, max, steps);
        self
    }

    pub fn scales(mut self, scales: Vec<f64>) -> Self {
        self.config.scale = ScaleConfig::explicit(scales);
        self
    }

    pub fn nms(mut self, config: NmsConfig) -> Self {
        self.config.nms = config;
        self
    }

    pub fn max_matches(mut self, max_matches: Option<usize>) -> Self {
        self.config.max_matches = max_matches;
        self
    }

    pub fn per_scale_max(mut self, max_matches: Option<usize>) -> Self {
        self.config.per_scale_max = max_matches;
        self
    }

    pub fn adaptive(mut self, adaptive: Option<AdaptiveConfig>) -> Self {
        self.config.adaptive = adaptive;
        self
    }

    pub fn build(self) -> Result<TemplateModel> {
        match self.mask {
            Some(mask) => TemplateModel::with_mask(self.template, mask, self.config),
            None => TemplateModel::new(self.template, self.config),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TemplateModel {
    threshold: f32,
    matching_method: i32,
    score_mode: ScoreMode,
    nms: NmsConfig,
    max_matches: Option<usize>,
    per_scale_max: Option<usize>,
    adaptive: Option<AdaptiveConfig>,
    levels: Vec<TemplateLevel>,
    source_template: cv::core::Mat,
    source_mask: Option<cv::core::Mat>,
}

#[derive(Debug, Clone)]
struct TemplateLevel {
    scale: f64,
    template: cv::core::Mat,
    mask: Option<cv::core::Mat>,
}

#[derive(Debug, Clone, Copy)]
enum ScoreMode {
    HigherBetter,
    LowerBetterNormed,
}

impl TemplateModel {
    pub fn builder(template: cv::core::Mat) -> TemplateModelBuilder {
        TemplateModelBuilder::new(template)
    }

    pub fn new(template: cv::core::Mat, config: TemplateModelConfig) -> Result<Self> {
        ensure!(!template.empty(), "template image is empty");
        let score_mode = score_mode(config.matching_method)?;
        let scales = config.scale.build_scales()?;
        let levels = build_levels(&template, None, &scales)?;

        Ok(Self {
            threshold: config.threshold,
            matching_method: config.matching_method,
            score_mode,
            nms: config.nms,
            max_matches: config.max_matches,
            per_scale_max: config.per_scale_max,
            adaptive: config.adaptive,
            levels,
            source_template: template,
            source_mask: None,
        })
    }

    pub fn with_mask(
        template: cv::core::Mat,
        mask: cv::core::Mat,
        config: TemplateModelConfig,
    ) -> Result<Self> {
        ensure!(!template.empty(), "template image is empty");
        ensure!(!mask.empty(), "mask image is empty");
        ensure!(mask.channels() == 1, "mask image must be single-channel");
        ensure!(
            mask.cols() == template.cols() && mask.rows() == template.rows(),
            "mask size must match template size"
        );
        ensure!(
            mask_supported(config.matching_method),
            "mask is not supported for this matching method"
        );

        let score_mode = score_mode(config.matching_method)?;
        let scales = config.scale.build_scales()?;
        let levels = build_levels(&template, Some(&mask), &scales)?;

        Ok(Self {
            threshold: config.threshold,
            matching_method: config.matching_method,
            score_mode,
            nms: config.nms,
            max_matches: config.max_matches,
            per_scale_max: config.per_scale_max,
            adaptive: config.adaptive,
            levels,
            source_template: template,
            source_mask: Some(mask),
        })
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn matching_method(&self) -> i32 {
        self.matching_method
    }

    pub fn match_all(&self, input: &cv::core::Mat) -> Result<Vec<TemplateMatch>> {
        if let Some(adaptive) = &self.adaptive {
            self.match_all_adaptive(input, adaptive.clone())
        } else {
            self.match_all_base(input, self.threshold)
        }
    }

    pub fn match_all_base(
        &self,
        input: &cv::core::Mat,
        threshold: f32,
    ) -> Result<Vec<TemplateMatch>> {
        ensure_compatible(input, &self.source_template)?;

        let matches = self
            .levels
            .par_iter()
            .try_fold(Vec::new, |mut acc, level| -> Result<Vec<TemplateMatch>> {
                if input.cols() < level.template.cols() || input.rows() < level.template.rows() {
                    return Ok(acc);
                }

                let result = run_match_template(
                    input,
                    &level.template,
                    level.mask.as_ref(),
                    self.matching_method,
                )?;

                let mut level_matches = collect_matches_from_result(
                    &result,
                    level.template.size()?,
                    level.scale,
                    threshold,
                    self.per_scale_max,
                    self.score_mode,
                )?;
                acc.append(&mut level_matches);
                Ok(acc)
            })
            .try_reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                Ok(a)
            })?;

        apply_nms_and_limit(matches, self.nms, self.max_matches)
    }

    pub fn match_all_raw(
        &self,
        input: &cv::core::Mat,
        threshold: f32,
    ) -> Result<Vec<TemplateMatch>> {
        ensure_compatible(input, &self.source_template)?;

        let matches = self
            .levels
            .par_iter()
            .try_fold(Vec::new, |mut acc, level| -> Result<Vec<TemplateMatch>> {
                if input.cols() < level.template.cols() || input.rows() < level.template.rows() {
                    return Ok(acc);
                }

                let result = run_match_template(
                    input,
                    &level.template,
                    level.mask.as_ref(),
                    self.matching_method,
                )?;

                let mut level_matches = collect_matches_from_result(
                    &result,
                    level.template.size()?,
                    level.scale,
                    threshold,
                    self.per_scale_max,
                    self.score_mode,
                )?;
                acc.append(&mut level_matches);
                Ok(acc)
            })
            .try_reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                Ok(a)
            })?;

        let mut matches = matches;
        matches.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(matches)
    }

    pub fn match_all_adaptive(
        &self,
        input: &cv::core::Mat,
        adaptive: AdaptiveConfig,
    ) -> Result<Vec<TemplateMatch>> {
        ensure_compatible(input, &self.source_template)?;
        ensure!(
            !adaptive.scales.is_empty(),
            "adaptive scales must not be empty"
        );

        let mut bootstrap = self.match_all_raw(input, self.threshold)?;
        if bootstrap.is_empty() {
            let fallback = (self.threshold * adaptive.fallback_ratio)
                .clamp(adaptive.fallback_min, adaptive.fallback_max);
            if fallback < self.threshold {
                bootstrap = self.match_all_raw(input, fallback)?;
            }
        }

        let best = match bootstrap.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Some(best) => best.clone(),
            None => return Ok(Vec::new()),
        };

        let cropped = crop_template_from_match(input, &best)?;
        let refined_mask = match &self.source_mask {
            Some(mask) => Some(resize_mask_to_scale(mask, best.scale)?),
            None => None,
        };

        let mut config = TemplateModelConfig::default();
        config.threshold = self.threshold;
        config.matching_method = self.matching_method;
        config.scale = ScaleConfig::explicit(adaptive.scales.clone());
        config.nms = self.nms;
        config.max_matches = self.max_matches;
        config.per_scale_max = self.per_scale_max;
        config.adaptive = None;

        let refined_model = match refined_mask {
            Some(mask) => TemplateModel::with_mask(cropped, mask, config)?,
            None => TemplateModel::new(cropped, config)?,
        };

        let mut refined = refined_model.match_all_base(input, self.threshold)?;
        for result in &mut refined {
            result.scale *= best.scale;
        }
        Ok(refined)
    }
}

fn ensure_compatible(input: &cv::core::Mat, template: &cv::core::Mat) -> Result<()> {
    ensure!(!input.empty(), "input image is empty");
    ensure!(
        input.depth() == template.depth(),
        "input depth must match template depth"
    );
    ensure!(
        input.channels() == template.channels(),
        "input channels must match template channels"
    );
    Ok(())
}

fn score_mode(method: i32) -> Result<ScoreMode> {
    match method {
        imgproc::TM_SQDIFF => bail!("TM_SQDIFF is not supported by TemplateModel v2"),
        imgproc::TM_SQDIFF_NORMED => Ok(ScoreMode::LowerBetterNormed),
        _ => Ok(ScoreMode::HigherBetter),
    }
}

fn mask_supported(method: i32) -> bool {
    matches!(method, imgproc::TM_SQDIFF | imgproc::TM_CCORR_NORMED)
}

fn build_levels(
    template: &cv::core::Mat,
    mask: Option<&cv::core::Mat>,
    scales: &[f64],
) -> Result<Vec<TemplateLevel>> {
    let levels = scales
        .par_iter()
        .filter_map(|&scale| {
            if scale <= 0.0 || !scale.is_finite() {
                return None;
            }

            let width = (template.cols() as f64 * scale).round() as i32;
            let height = (template.rows() as f64 * scale).round() as i32;
            if width < 1 || height < 1 {
                return None;
            }

            let template_res = if (scale - 1.0).abs() <= TOLERANCE {
                Ok(template.clone())
            } else {
                resize_template(template, width, height)
            };

            let mask_res = if let Some(mask) = mask {
                if (scale - 1.0).abs() <= TOLERANCE {
                    Ok(Some(mask.clone()))
                } else {
                    resize_mask(mask, width, height).map(Some)
                }
            } else {
                Ok(None)
            };

            match (template_res, mask_res) {
                (Ok(template_res), Ok(mask_res)) => Some(Ok(TemplateLevel {
                    scale,
                    template: template_res,
                    mask: mask_res,
                })),
                (Err(e), _) => Some(Err(e)),
                (_, Err(e)) => Some(Err(e)),
            }
        })
        .collect::<Result<Vec<_>>>()?;

    let mut levels = levels;
    levels.sort_by(|a, b| {
        a.scale
            .partial_cmp(&b.scale)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ensure!(!levels.is_empty(), "no valid pyramid levels generated");
    Ok(levels)
}

fn resize_template(template: &cv::core::Mat, width: i32, height: i32) -> Result<cv::core::Mat> {
    ensure!(width > 0 && height > 0, "scaled template size is invalid");
    let scale = width as f64 / template.cols() as f64;
    let interpolation = if scale >= 1.0 {
        imgproc::INTER_LINEAR
    } else {
        imgproc::INTER_AREA
    };

    let mut resized = cv::core::Mat::default();
    imgproc::resize(
        template,
        &mut resized,
        cv::core::Size::new(width, height),
        0.0,
        0.0,
        interpolation,
    )?;
    Ok(resized)
}

fn resize_mask(mask: &cv::core::Mat, width: i32, height: i32) -> Result<cv::core::Mat> {
    ensure!(width > 0 && height > 0, "scaled mask size is invalid");
    let mut resized = cv::core::Mat::default();
    imgproc::resize(
        mask,
        &mut resized,
        cv::core::Size::new(width, height),
        0.0,
        0.0,
        imgproc::INTER_NEAREST,
    )?;
    Ok(resized)
}

fn resize_mask_to_scale(mask: &cv::core::Mat, scale: f64) -> Result<cv::core::Mat> {
    let width = (mask.cols() as f64 * scale).round() as i32;
    let height = (mask.rows() as f64 * scale).round() as i32;
    resize_mask(mask, width, height)
}

fn run_match_template(
    input: &cv::core::Mat,
    template: &cv::core::Mat,
    mask: Option<&cv::core::Mat>,
    method: i32,
) -> Result<cv::core::Mat> {
    let mut res = cv::core::Mat::default();
    if let Some(mask) = mask {
        imgproc::match_template(input, template, &mut res, method, mask)?;
    } else {
        imgproc::match_template(input, template, &mut res, method, &cv::core::no_array())?;
    }
    Ok(res)
}

fn collect_matches_from_result(
    result: &cv::core::Mat,
    template_size: cv::core::Size,
    scale: f64,
    threshold: f32,
    per_scale_max: Option<usize>,
    score_mode: ScoreMode,
) -> Result<Vec<TemplateMatch>> {
    let mut result = if result.depth() == CV_32F {
        result.clone()
    } else {
        let mut converted = cv::core::Mat::default();
        result.convert_to(&mut converted, CV_32F, 1.0, 0.0)?;
        converted
    };

    if !result.is_continuous() {
        let mut continuous = cv::core::Mat::default();
        result.copy_to(&mut continuous)?;
        result = continuous;
    }

    let rows = result.rows() as usize;
    let cols = result.cols() as usize;
    let step = result.step1_def()? as usize;
    let data = result.data_typed::<f32>()?;

    let mut matches = Vec::new();
    for y in 0..rows {
        let row = y * step;
        for x in 0..cols {
            let value = data[row + x];
            let score = match score_mode {
                ScoreMode::HigherBetter => value,
                ScoreMode::LowerBetterNormed => 1.0 - value,
            };
            if score >= threshold {
                matches.push(TemplateMatch {
                    position: cv::core::Point::new(x as i32, y as i32),
                    size: template_size,
                    score,
                    scale,
                });
            }
        }
    }

    if let Some(limit) = per_scale_max {
        if matches.len() > limit {
            matches.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            matches.truncate(limit);
        }
    }

    Ok(matches)
}

fn apply_nms_and_limit(
    matches: Vec<TemplateMatch>,
    nms_config: NmsConfig,
    max_matches: Option<usize>,
) -> Result<Vec<TemplateMatch>> {
    if matches.is_empty() {
        return Ok(matches);
    }

    let keep = TemplateMatch::calc_nms_indices(&matches, nms_config);
    let mut filtered = keep.iter().map(|&i| matches[i].clone()).collect::<Vec<_>>();
    filtered.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(limit) = max_matches {
        filtered.truncate(limit);
    }
    Ok(filtered)
}

fn crop_template_from_match(src: &cv::core::Mat, result: &TemplateMatch) -> Result<cv::core::Mat> {
    let size = src.size()?;
    let mut rect = cv::core::Rect::new(
        result.position.x,
        result.position.y,
        result.size.width,
        result.size.height,
    );

    if rect.x < 0 {
        rect.width -= -rect.x;
        rect.x = 0;
    }
    if rect.y < 0 {
        rect.height -= -rect.y;
        rect.y = 0;
    }
    if rect.x + rect.width > size.width {
        rect.width = size.width - rect.x;
    }
    if rect.y + rect.height > size.height {
        rect.height = size.height - rect.y;
    }
    ensure!(
        rect.width > 0 && rect.height > 0,
        "cropped template is out of bounds"
    );

    let roi = src.roi(rect)?;
    Ok(roi.clone_pointee())
}
