use anyhow::{ensure, Result};
use ndarray as nd;
use opencv::core::{self as cv, MatTraitConst};
use opencv::imgproc;

use crate::nms::nms;

use super::NmsConfig;
pub(crate) const MIN_AREA: i32 = 128;
pub(crate) const CANDIDATE: usize = 5;
pub(crate) const INVALID: f64 = -1.0;

#[derive(Debug, Clone, Copy)]
pub struct Pose {
    /// Top-left anchor point of the matched template.
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub angle: f32,
    pub score: f32,
}

impl Pose {
    pub fn nms_filter(poses: &[Pose], nms_config: NmsConfig) -> Result<Vec<Pose>> {
        let keep = Self::nms_filter_indices(poses, nms_config)?;
        Ok(keep.iter().map(|&idx| poses[idx]).collect())
    }

    pub(crate) fn nms_filter_indices(poses: &[Pose], nms_config: NmsConfig) -> Result<Vec<usize>> {
        if poses.is_empty() {
            return Ok(Vec::new());
        }

        let (boxes, scores) = Self::nms_inputs(poses)?;
        if boxes.nrows() == 0 {
            return Ok(Vec::new());
        }

        Ok(nms(
            &boxes,
            &scores,
            nms_config.iou_threshold,
            nms_config.score_threshold,
        ))
    }

    pub(crate) fn nms_inputs(poses: &[Pose]) -> Result<(nd::Array2<i32>, nd::Array1<f64>)> {
        let mut boxes = nd::Array2::<i32>::default((0, 4));
        let mut scores = Vec::with_capacity(poses.len());
        for pose in poses {
            let rect = pose_to_box(pose)?;
            boxes.push(nd::Axis(0), nd::ArrayView::from(&rect)).unwrap();
            scores.push(pose.score as f64);
        }
        Ok((boxes, nd::Array1::from(scores)))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScaledPose {
    pub pose: Pose,
    pub scale: f64,
}

impl ScaledPose {
    pub fn nms_filter(poses: &[ScaledPose], nms_config: NmsConfig) -> Result<Vec<ScaledPose>> {
        let raw_poses: Vec<Pose> = poses.iter().map(|pose| pose.pose).collect();
        let keep = Pose::nms_filter_indices(&raw_poses, nms_config)?;
        Ok(keep.iter().map(|&idx| poses[idx]).collect())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    pub(crate) pos: cv::Point2d,
    pub(crate) angle: f64,
    pub(crate) score: f64,
}

impl Candidate {
    pub(crate) fn new(pos: cv::Point2d, angle: f64, score: f64) -> Self {
        Self { pos, angle, score }
    }

    pub(crate) fn filter_overlap(
        candidates: &mut [Candidate],
        rects: &[cv::RotatedRect],
        max_overlap: f64,
    ) -> Result<()> {
        let size = candidates.len();
        for i in 0..size {
            if candidates[i].score < 0.0 {
                continue;
            }

            for j in (i + 1)..size {
                if candidates[j].score < 0.0 {
                    continue;
                }

                let mut points = cv::Vector::<cv::Point2f>::new();
                let intersect_type =
                    imgproc::rotated_rectangle_intersection(rects[i], rects[j], &mut points)?;

                match intersect_type {
                    imgproc::INTERSECT_NONE => continue,
                    imgproc::INTERSECT_FULL => {
                        if candidates[i].score > candidates[j].score {
                            candidates[j].score = INVALID;
                        } else {
                            candidates[i].score = INVALID;
                        }
                    }
                    imgproc::INTERSECT_PARTIAL => {
                        if points.len() < 2 {
                            continue;
                        }
                        let area = imgproc::contour_area_def(&points)?;
                        let rect_area = rects[i].size.width as f64 * rects[i].size.height as f64;
                        if rect_area <= 0.0 {
                            continue;
                        }
                        let overlap = area / rect_area;
                        if overlap > max_overlap {
                            if candidates[i].score > candidates[j].score {
                                candidates[j].score = INVALID;
                            } else {
                                candidates[i].score = INVALID;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct Block {
    rect: cv::Rect,
    max_pos: cv::Point,
    max_score: f64,
}

impl Block {
    fn new(rect: cv::Rect, max_pos: cv::Point, max_score: f64) -> Self {
        Self {
            rect,
            max_pos,
            max_score,
        }
    }
}

pub(crate) struct BlockMax {
    blocks: Vec<Block>,
    pub(crate) score: cv::Mat,
}

impl BlockMax {
    pub(crate) fn new(score: cv::Mat, template_size: cv::Size) -> Result<Self> {
        let mut blocks = Vec::new();
        let block_width = template_size.width * 2;
        let block_height = template_size.height * 2;

        let score_size = score.size()?;
        let n_width = score_size.width / block_width;
        let n_height = score_size.height / block_height;
        let h_remained = score_size.width % block_width;
        let v_remained = score_size.height % block_height;

        for y in 0..n_height {
            for x in 0..n_width {
                let rect =
                    cv::Rect::new(x * block_width, y * block_height, block_width, block_height);
                let mut max_score = 0.0;
                let mut max_pos = cv::Point::new(0, 0);
                let roi = score.roi(rect)?;
                cv::min_max_loc(
                    &roi,
                    None,
                    Some(&mut max_score),
                    None,
                    Some(&mut max_pos),
                    &cv::no_array(),
                )?;
                max_pos.x += rect.x;
                max_pos.y += rect.y;
                blocks.push(Block::new(rect, max_pos, max_score));
            }
        }

        if h_remained > 0 {
            let rect = cv::Rect::new(n_width * block_width, 0, h_remained, score_size.height);
            let mut max_score = 0.0;
            let mut max_pos = cv::Point::new(0, 0);
            let roi = score.roi(rect)?;
            cv::min_max_loc(
                &roi,
                None,
                Some(&mut max_score),
                None,
                Some(&mut max_pos),
                &cv::no_array(),
            )?;
            max_pos.x += rect.x;
            max_pos.y += rect.y;
            blocks.push(Block::new(rect, max_pos, max_score));
        }

        if v_remained > 0 {
            let width = if h_remained > 0 {
                n_width * block_width
            } else {
                score_size.width
            };
            if width > 0 {
                let rect = cv::Rect::new(0, n_height * block_height, width, v_remained);
                let mut max_score = 0.0;
                let mut max_pos = cv::Point::new(0, 0);
                let roi = score.roi(rect)?;
                cv::min_max_loc(
                    &roi,
                    None,
                    Some(&mut max_score),
                    None,
                    Some(&mut max_pos),
                    &cv::no_array(),
                )?;
                max_pos.x += rect.x;
                max_pos.y += rect.y;
                blocks.push(Block::new(rect, max_pos, max_score));
            }
        }

        Ok(Self { blocks, score })
    }

    pub(crate) fn update(&mut self, rect: cv::Rect) -> Result<()> {
        for block in &mut self.blocks {
            if rect_intersection(block.rect, rect).is_none() {
                continue;
            }

            let mut max_score = 0.0;
            let mut max_pos = cv::Point::new(0, 0);
            let roi = self.score.roi(block.rect)?;
            cv::min_max_loc(
                &roi,
                None,
                Some(&mut max_score),
                None,
                Some(&mut max_pos),
                &cv::no_array(),
            )?;
            max_pos.x += block.rect.x;
            max_pos.y += block.rect.y;
            block.max_score = max_score;
            block.max_pos = max_pos;
        }

        Ok(())
    }

    pub(crate) fn max_value_loc(&self) -> Option<(f64, cv::Point)> {
        self.blocks
            .iter()
            .max_by(|a, b| {
                a.max_score
                    .partial_cmp(&b.max_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|block| (block.max_score, block.max_pos))
    }
}

pub(crate) fn compute_layers(width: i32, height: i32, min_area: i32) -> i32 {
    let mut area = width * height;
    let mut layer = 0;
    while area > min_area {
        area /= 4;
        layer += 1;
    }
    layer
}

pub(crate) fn size_area(size: cv::Size) -> i32 {
    size.width * size.height
}

pub(crate) fn pose_to_box(pose: &Pose) -> Result<[i32; 4]> {
    ensure!(
        pose.width > 0.0 && pose.height > 0.0,
        "pose size is invalid"
    );
    let angle = pose.angle as f64 * std::f64::consts::PI / 180.0;
    let cos = angle.cos();
    let sin = angle.sin();
    let x0 = pose.x as f64;
    let y0 = pose.y as f64;
    let w = pose.width as f64;
    let h = pose.height as f64;

    let points = [
        (0.0, 0.0),
        (w, 0.0),
        (w, h),
        (0.0, h),
    ];

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (dx, dy) in points {
        let x = x0 + dx * cos - dy * sin;
        let y = y0 + dx * sin + dy * cos;
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    Ok([
        min_x.floor() as i32,
        min_y.floor() as i32,
        max_x.ceil() as i32,
        max_y.ceil() as i32,
    ])
}

fn rect_intersection(a: cv::Rect, b: cv::Rect) -> Option<cv::Rect> {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);
    if x2 > x1 && y2 > y1 {
        Some(cv::Rect::new(x1, y1, x2 - x1, y2 - y1))
    } else {
        None
    }
}
