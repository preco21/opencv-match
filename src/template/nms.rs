use ndarray as nd;

pub fn nms(
    boxes: &nd::Array2<i32>,
    scores: &nd::Array1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> Vec<usize> {
    assert_eq!(boxes.nrows(), scores.len());

    let mut order: Vec<usize> = if score_threshold > 0.0 {
        scores
            .iter()
            .enumerate()
            .filter(|(_, &score)| score >= score_threshold)
            .map(|(idx, _)| idx)
            .collect()
    } else {
        (0..scores.len()).collect()
    };

    order.sort_unstable_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep: Vec<usize> = Vec::new();
    let mut suppress = vec![false; order.len()];

    for (i, &idx) in order.iter().enumerate() {
        if suppress[i] {
            continue;
        }
        keep.push(idx);
        let box1 = boxes.row(idx);
        let b1x = box1[0];
        let b1y = box1[1];
        let b1xx = box1[2];
        let b1yy = box1[3];
        let area1 = ((b1xx - b1x) * (b1yy - b1y)) as f64;
        for j in (i + 1)..order.len() {
            if suppress[j] {
                continue;
            }
            let box2 = boxes.row(order[j]);
            let b2x = box2[0];
            let b2y = box2[1];
            let b2xx = box2[2];
            let b2yy = box2[3];

            let x = if b1x > b2x { b1x } else { b2x };
            let y = if b1y > b2y { b1y } else { b2y };
            let xx = if b1xx < b2xx { b1xx } else { b2xx };
            let yy = if b1yy < b2yy { b1yy } else { b2yy };
            if x > xx || y > yy {
                continue;
            }

            let intersection = ((xx - x) * (yy - y)) as f64;
            let area2 = ((b2xx - b2x) * (b2yy - b2y)) as f64;
            let union = area1 + area2 - intersection;
            let iou = intersection / union;
            if iou > iou_threshold {
                suppress[j] = true;
            }
        }
    }

    keep
}
