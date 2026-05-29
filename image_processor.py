# image_processor.py

import cv2
import numpy as np
import time
from datetime import datetime
import os

import config
from calibration import get_mm_per_pixel

# Edge detection configuration (from config.py)
EDGE_CANNY_LOW = config.EDGE_CANNY_LOW
EDGE_CANNY_HIGH = config.EDGE_CANNY_HIGH
EDGE_BLUR_KERNEL = config.EDGE_BLUR_KERNEL
EDGE_DILATE_KERNEL = config.EDGE_DILATE_KERNEL
EDGE_ROI_TOP_FRACTION = config.EDGE_ROI_TOP_FRACTION
EDGE_ROI_BOTTOM_FRACTION = config.EDGE_ROI_BOTTOM_FRACTION
EDGE_ROI_LEFT_FRACTION = config.EDGE_ROI_LEFT_FRACTION
EDGE_ROI_RIGHT_FRACTION = config.EDGE_ROI_RIGHT_FRACTION
EDGE_ENVELOPE_SMOOTH_KERNEL = config.EDGE_ENVELOPE_SMOOTH_KERNEL

class ImageProcessor:
    def __init__(self, model):
        """
        Initializes the ImageProcessor object with the given model.

        Args:
            model (object): The YOLO model used for stitch and edge detection.

        Attributes:
            model (object): The YOLO model used for stitch and edge detection.
            mm_per_pixel (float): The conversion factor from pixels to millimeters.
            last_processed_time (float): The timestamp of the last processed frame (epoch seconds).

            last_avg_stitch_length_mm (float|None): latest measurable stitch length for DB logging.
            last_avg_stitch_edge_distance_mm (float|None): latest measurable seam allowance for DB logging.
        """
        self.model = model
        self.mm_per_pixel = get_mm_per_pixel()
        self.last_processed_time = 0.0

        # ✅ Latest values to be inserted into DB by mysql thread
        self.last_avg_stitch_length_mm = None
        self.last_avg_stitch_edge_distance_mm = None

    def _is_stitch_length_in_ideal_range(self, value_mm):
        """Return True when stitch length is inside configured ideal limits."""
        if value_mm is None:
            return False
        return config.IDEAL_STITCH_LENGTH_MM_MIN <= value_mm <= config.IDEAL_STITCH_LENGTH_MM_MAX

    def _is_seam_allowance_in_ideal_range(self, value_mm):
        """Return True when seam allowance is inside configured ideal limits."""
        if value_mm is None:
            return False
        return config.IDEAL_SEAM_ALLOWANCE_MM_MIN <= value_mm <= config.IDEAL_SEAM_ALLOWANCE_MM_MAX

    def _closest_point_on_segment(self, px, py, ax, ay, bx, by):
        vx = bx - ax
        vy = by - ay
        seg_len2 = vx * vx + vy * vy
        if seg_len2 == 0.0:
            dx = px - ax
            dy = py - ay
            return float(ax), float(ay), dx * dx + dy * dy
        t = ((px - ax) * vx + (py - ay) * vy) / seg_len2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        cx = ax + t * vx
        cy = ay + t * vy
        dx = px - cx
        dy = py - cy
        return float(cx), float(cy), dx * dx + dy * dy

    def _closest_point_on_polyline(self, px, py, points):
        if points is None or len(points) < 2:
            return None
        best = None
        best_dist2 = None
        for i in range(len(points) - 1):
            ax, ay = points[i]
            bx, by = points[i + 1]
            cx, cy, dist2 = self._closest_point_on_segment(px, py, ax, ay, bx, by)
            if best_dist2 is None or dist2 < best_dist2:
                best_dist2 = dist2
                best = (cx, cy, dist2)
        return best

    def get_perpendicular_distance_to_edges(self, centroid, mask):
        """Calculate perpendicular distances from a centroid to top and bottom mask edges"""
        binary_mask = mask.astype(np.uint8)
        h, w = binary_mask.shape
        cx, cy = centroid

        top_distance = float('inf')
        bottom_distance = float('inf')
        top_point = None
        bottom_point = None

        cx = int(cx)
        cy = int(cy)

        # Scan up from centroid to find first transition (edge)
        for y in range(cy, -1, -1):
            if 0 <= y < h and (y + 1) < h and 0 <= cx < w:
                if binary_mask[y, cx] == 0 and binary_mask[y + 1, cx] == 1:
                    top_distance = cy - y
                    top_point = (cx, y)
                    break

        # Scan down from centroid to find first transition (edge)
        for y in range(cy, h):
            if 0 <= y < h and (y - 1) >= 0 and 0 <= cx < w:
                if binary_mask[y, cx] == 0 and binary_mask[y - 1, cx] == 1:
                    bottom_distance = y - cy
                    bottom_point = (cx, y)
                    break

        return top_distance, top_point, bottom_distance, bottom_point

    def calculate_stitch_edge_distances(self, result):
        """Calculate the distance between stitches and edge using segmentation masks"""
        stitch_centers = []
        edge_centers = []
        min_required = getattr(config, "MIN_STITCH_DETECTIONS", 3)

        # Mask dimensions
        if hasattr(result, 'orig_img') and result.orig_img is not None:
            mask_h, mask_w = result.orig_img.shape[:2]
        else:
            mask_h, mask_w = config.FRAME_H, config.FRAME_W

        # Collect stitch centers and select the highest-confidence edge center
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()

            # ROI bounds from config to avoid corner artifacts
            roi_x1 = EDGE_ROI_LEFT_FRACTION
            roi_x2 = EDGE_ROI_RIGHT_FRACTION
            roi_y1 = EDGE_ROI_TOP_FRACTION
            roi_y2 = EDGE_ROI_BOTTOM_FRACTION

            best_edge_idx = None
            best_edge_conf = -1.0
            best_edge_center = None

            for i, (x1, y1, x2, y2) in enumerate(boxes):
                if confidence[i] >= 0.25:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    if int(classes[i]) == config.STITCH_CLASS_ID:
                        # Ignore boxes whose center is outside the central ROI
                        if not (roi_x1 <= center_x / mask_w <= roi_x2 and roi_y1 <= center_y / mask_h <= roi_y2):
                            continue
                        stitch_centers.append((center_x, center_y))
                    elif int(classes[i]) == config.EDGE_CLASS_ID:
                        # Ignore boxes whose center is outside the central ROI
                        if not (roi_x1 <= center_x / mask_w <= roi_x2 and roi_y1 <= center_y / mask_h <= roi_y2):
                            continue
                        if confidence[i] > best_edge_conf:
                            best_edge_conf = confidence[i]
                            best_edge_idx = i
                            best_edge_center = (center_x, center_y)

            if best_edge_center is not None:
                edge_centers.append(best_edge_center)

        segmentation_contour = None
        combined_edge_mask = None

        # Build a mask from the highest-confidence edge detection
        if hasattr(result, 'masks') and result.masks is not None and result.boxes is not None:
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()

            if best_edge_idx is not None and best_edge_idx < len(masks) and int(classes[best_edge_idx]) == config.EDGE_CLASS_ID and confidence[best_edge_idx] >= 0.3:
                mask_resized = cv2.resize(
                    masks[best_edge_idx].astype(np.float32),
                    (mask_w, mask_h),
                    interpolation=cv2.INTER_LINEAR
                )
                combined_edge_mask = mask_resized > 0.5

        if combined_edge_mask is None:
            return {
                'stitch_centers': stitch_centers,
                'edge_centers': edge_centers,
                'edge_y_line': None,
                'edge_line_points': None,
                'segmentation_contour': None,
                'all_distances': [],
                'avg_distance_mm': None
            }

        mask_uint8 = (combined_edge_mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            segmentation_contour = max(contours, key=cv2.contourArea)
            contour_points = segmentation_contour.reshape(-1, 2)
        else:
            contour_points = None

        if contour_points is None or contour_points.size == 0 or len(stitch_centers) < min_required:
            return {
                'stitch_centers': stitch_centers,
                'edge_centers': edge_centers,
                'edge_y_line': None,
                'edge_line_points': None,
                'segmentation_contour': segmentation_contour,
                'all_distances': [],
                'avg_distance_mm': None
            }

        # Simplify contour for efficiency and stable normals.
        arc_len = cv2.arcLength(segmentation_contour, True)
        approx_eps = max(1.0, 0.002 * arc_len)
        approx = cv2.approxPolyDP(segmentation_contour, approx_eps, True)
        contour_points = approx.reshape(-1, 2).astype(np.float32)
        contour_poly = [(float(x), float(y)) for x, y in contour_points]
        if len(contour_poly) < 2:
            return {
                'stitch_centers': stitch_centers,
                'edge_centers': edge_centers,
                'edge_y_line': None,
                'edge_line_points': None,
                'segmentation_contour': segmentation_contour,
                'all_distances': [],
                'avg_distance_mm': None
            }
        if contour_poly[0] != contour_poly[-1]:
            contour_poly.append(contour_poly[0])

        all_distances = []
        total_distance_mm = 0.0
        valid_distance_count = 0

        max_steps = int(getattr(config, "SEG_OUTER_EDGE_MAX_STEPS", max(mask_w, mask_h)))
        max_steps = max(1, min(max_steps, int(max(mask_w, mask_h))))
        for stitch_center in stitch_centers:
            cx, cy = float(stitch_center[0]), float(stitch_center[1])
            closest = self._closest_point_on_polyline(cx, cy, contour_poly)
            if closest is None:
                continue
            inner_x, inner_y, _ = closest

            vec_x = inner_x - cx
            vec_y = inner_y - cy
            vec_norm = float(np.hypot(vec_x, vec_y))
            if vec_norm == 0.0:
                continue

            dir_x = vec_x / vec_norm
            dir_y = vec_y / vec_norm
            outer_x, outer_y = float(inner_x), float(inner_y)

            for _ in range(max_steps):
                nx = outer_x + dir_x
                ny = outer_y + dir_y
                ix = int(round(nx))
                iy = int(round(ny))
                if ix < 0 or ix >= mask_w or iy < 0 or iy >= mask_h:
                    break
                if combined_edge_mask[iy, ix]:
                    outer_x, outer_y = nx, ny
                else:
                    break

            edge_point = (int(round(outer_x)), int(round(outer_y)))
            distance_pixels = float(np.hypot(outer_x - cx, outer_y - cy))
            distance_mm = distance_pixels * self.mm_per_pixel
            total_distance_mm += distance_mm
            valid_distance_count += 1

            all_distances.append({
                'stitch_center': stitch_center,
                'edge_point': edge_point,
                'edge_y': edge_point[1],
                'distance_pixels': distance_pixels,
                'distance_mm': distance_mm
            })

        avg_distance_mm = total_distance_mm / valid_distance_count if valid_distance_count > 0 else None

        return {
            'stitch_centers': stitch_centers,
            'edge_centers': edge_centers,
            'edge_y_line': None,
            'edge_line_points': None,
            'segmentation_contour': segmentation_contour,
            'all_distances': all_distances,
            'avg_distance_mm': avg_distance_mm
        }

    def detect_fabric_edge_canny(self, frame, canny_low=EDGE_CANNY_LOW, canny_high=EDGE_CANNY_HIGH,
                                blur_ksize=EDGE_BLUR_KERNEL, dilate_ksize=EDGE_DILATE_KERNEL,
                                roi_top_frac=EDGE_ROI_TOP_FRACTION,
                                roi_bottom_frac=EDGE_ROI_BOTTOM_FRACTION,
                                roi_left_frac=EDGE_ROI_LEFT_FRACTION,
                                roi_right_frac=EDGE_ROI_RIGHT_FRACTION,
                                smooth_ksize=EDGE_ENVELOPE_SMOOTH_KERNEL):
        """Detect fabric edge using Canny edge detection and return the rightmost envelope.

        Strategy:
            1. Convert to grayscale and blur to reduce noise.
            2. Apply Canny edge detection.
            3. Optionally dilate to connect nearby edge fragments.
            4. Restrict search to a rectangular ROI defined by fractional bounds
            (width: roi_left_frac to roi_right_frac, height: roi_top_frac to roi_bottom_frac).
            5. For each row inside the ROI, find the rightmost edge pixel — this traces the fabric edge.
            6. Smooth the resulting envelope with a median filter.

        Args:
            frame: Input BGR image from camera.
            canny_low: Lower threshold for Canny.
            canny_high: Upper threshold for Canny.
            blur_ksize: Gaussian blur kernel size (odd number).
            dilate_ksize: Dilation kernel size to connect nearby edges (0 = skip).
            roi_top_frac: Top boundary of ROI as fraction of image height (0.0–1.0).
            roi_bottom_frac: Bottom boundary of ROI as fraction of image height (0.0–1.0).
            roi_left_frac: Left boundary of ROI as fraction of image width (0.0–1.0).
            roi_right_frac: Right boundary of ROI as fraction of image width (0.0–1.0).
            smooth_ksize: Kernel size for median smoothing of the envelope (odd, 0 = skip).

        Returns:
                envelope: 1D int array of length h. envelope[y] = x-coordinate of the
                    detected fabric edge in row y, or -1 if no edge found.
            edge_map: Binary edge image (useful for visualization / debugging).
            roi_rect: Tuple (roi_x1, roi_y1, roi_x2, roi_y2) pixel coordinates of the ROI rectangle.
        """
        h, w = frame.shape[:2]

        # 1. Grayscale + blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if blur_ksize > 0:
            ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
            gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # 2. Canny edge detection
        edges = cv2.Canny(gray, canny_low, canny_high)

        # 3. Optional dilation to bridge small gaps
        if dilate_ksize > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ksize, dilate_ksize))
            edges = cv2.dilate(edges, kernel, iterations=1)

        # 3.5. Filter contours to keep only lengthy edges, discard short ones
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_length = 100  # Minimum contour length to keep
        long_contours = [c for c in contours if cv2.arcLength(c, False) > min_length]
        edges = np.zeros_like(edges)
        if long_contours:
            cv2.drawContours(edges, long_contours, -1, 255, 3)  # Thicker lines

        # 4. Mask out everything OUTSIDE the rectangular ROI
        roi_y1 = max(0, min(int(h * roi_top_frac), h - 1))
        roi_y2 = max(roi_y1 + 1, min(int(h * roi_bottom_frac), h))
        roi_x1 = max(0, min(int(w * roi_left_frac), w - 1))
        roi_x2 = max(roi_x1 + 1, min(int(w * roi_right_frac), w))

        mask = np.zeros_like(edges)
        mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255
        edges = cv2.bitwise_and(edges, mask)

        roi_rect = (roi_x1, roi_y1, roi_x2, roi_y2)

        # 5. For each row, find the RIGHTMOST edge pixel (rightmost envelope)
        envelope = np.full((h,), -1, dtype=int)  # Now indexed by ROW, not column
        rev = edges[:, ::-1]              # Horizontal flip
        has_any = rev.any(axis=1)        # Check each ROW
        idx_in_rev = np.argmax(rev > 0, axis=1)

        for y in range(h):
            if has_any[y]:
                envelope[y] = w - 1 - idx_in_rev[y]  # Rightmost x in this row

        # 6. Smooth the envelope with a median filter to remove noise
        if smooth_ksize > 0:
            ksize = smooth_ksize if smooth_ksize % 2 == 1 else smooth_ksize + 1
            valid_mask = envelope >= 0
            if valid_mask.sum() > ksize:
                # Only smooth valid entries; keep -1 for invalid
                temp = envelope.astype(np.float32).copy()
                temp[~valid_mask] = np.nan
                # Fill NaN gaps with nearest valid for filtering, then restore
                filled = temp.copy()
                # Forward fill
                for i in range(1, h):
                    if np.isnan(filled[i]) and not np.isnan(filled[i-1]):
                        filled[i] = filled[i-1]
                # Backward fill
                for i in range(h-2, -1, -1):
                    if np.isnan(filled[i]) and not np.isnan(filled[i+1]):
                        filled[i] = filled[i+1]

                if not np.isnan(filled).all():
                    filled = np.nan_to_num(filled, nan=0.0).astype(int)
                    # cv2.medianBlur only supports uint8, but envelope values can exceed 255.
                    # Use a manual sliding-window median instead.
                    half_k = ksize // 2
                    smoothed = filled.copy()
                    for i in range(half_k, h - half_k):
                        smoothed[i] = int(np.median(filled[i - half_k : i + half_k + 1]))
                    # Restore invalids
                    envelope[valid_mask] = smoothed[valid_mask]

        return envelope, edges, roi_rect

    def calculate_stitch_edge_distances_canny(self, result):
        """
        Calculate the distance between stitches and edge using
        Canny-based rightmost envelope detection.
        """
        stitch_centers = []
        min_required = getattr(config, "MIN_STITCH_DETECTIONS", 3)
        
        # 0. Prepare frame dimensions (needed for ROI filtering)
        frame = result.orig_img
        h, w = frame.shape[:2]

        # ROI bounds from config.
        roi_x1 = EDGE_ROI_LEFT_FRACTION
        roi_x2 = EDGE_ROI_RIGHT_FRACTION
        roi_y1 = EDGE_ROI_TOP_FRACTION
        roi_y2 = EDGE_ROI_BOTTOM_FRACTION

        # 1. Collect stitch centers from YOLO/Object Detection boxes
        #    Only keep boxes inside the central ROI (middle 50% width, middle 50% height)
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()

            for i, (x1, y1, x2, y2) in enumerate(boxes):
                if confidence[i] >= 0.3 and int(classes[i]) == config.STITCH_CLASS_ID:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Ignore boxes whose center is outside the central ROI
                    if not (roi_x1 <= center_x / w <= roi_x2 and roi_y1 <= center_y / h <= roi_y2):
                        continue

                    stitch_centers.append((center_x, center_y))

        if len(stitch_centers) < min_required:
            return {
                'stitch_centers': stitch_centers,
                'edge_centers': [],
                'edge_map': None,
                'edge_line_points': None,
                'all_distances': [],
                'avg_distance_mm': None
            }

        # 2. Run Canny Edge Detection to get the fabric boundary (envelope)
        # Use your custom function with a restricted ROI to ignore corner edges
        envelope, edge_map, _ = self.detect_fabric_edge_canny(
            frame,
            roi_top_frac=roi_y1,
            roi_bottom_frac=roi_y2,
            roi_left_frac=roi_x1,
            roi_right_frac=roi_x2,
        )

        edge_line_points = [(int(x), int(y)) for y, x in enumerate(envelope) if x != -1]
        if len(edge_line_points) < 2:
            edge_line_points = None

        all_distances = []
        total_distance_mm = 0.0
        valid_distance_count = 0

        # 3. Calculate distances using the envelope
        for cx, cy in stitch_centers:
            ix, iy = int(cx), int(cy)
            
            # Ensure coordinates are within image bounds
            if 0 <= ix < w and 0 <= iy < h and edge_line_points is not None:
                closest = self._closest_point_on_polyline(cx, cy, edge_line_points)
                if closest is None:
                    continue
                edge_x, edge_y, dist2 = closest
                distance_pixels = float(np.sqrt(dist2))
                distance_mm = distance_pixels * self.mm_per_pixel
                edge_point = (int(round(edge_x)), int(round(edge_y)))

                total_distance_mm += distance_mm
                valid_distance_count += 1

                all_distances.append({
                    'stitch_center': (cx, cy),
                    'edge_x': float(edge_x),
                    'edge_point': edge_point,
                    'distance_pixels': float(distance_pixels),
                    'distance_mm': float(distance_mm)
                })

        # 4. Final aggregation
        avg_distance_mm = total_distance_mm / valid_distance_count if valid_distance_count > 0 else None

        # Optional: Logic to handle cases where no edges were found in stitch columns
        if avg_distance_mm is None:
            # If we detected an edge envelope anywhere, use its mean x as a fallback edge line.
            edge_xs = [x for x in envelope if x != -1]
            if edge_xs and stitch_centers:
                fallback_edge_x = float(np.mean(edge_xs))
                for cx, cy in stitch_centers:
                    distance_pixels = abs(cx - fallback_edge_x)
                    distance_mm = distance_pixels * self.mm_per_pixel
                    all_distances.append({
                        'stitch_center': (cx, cy),
                        'edge_x': fallback_edge_x,
                        'edge_point': (int(round(fallback_edge_x)), int(round(cy))),
                        'distance_pixels': float(distance_pixels),
                        'distance_mm': float(distance_mm)
                    })
                    total_distance_mm += distance_mm
                valid_distance_count = len(stitch_centers)
                avg_distance_mm = total_distance_mm / valid_distance_count
                # print("[WARNING] No edge found in stitch columns: using mean envelope y fallback for distance calculation.")
            else:
                pass
                # print("[WARNING] No fabric edge detected in columns containing stitches.")

        # Debug logging: why avg_distance_mm might be None (suppressed - only print if needed for troubleshooting)
        # stitch_count = len(stitch_centers)
        # edge_columns = sum(1 for y in envelope if y != -1)
        # calculated_distances = len(all_distances)
        # print(f"[DEBUG] stitch_centers={stitch_count}, edge_columns={edge_columns}, distances_computed={calculated_distances}, avg_distance_mm={avg_distance_mm}")

        # Provide a minimal edge_centers list so caller code can report an edge count.
        # We use a representative point (mean edge x, center y) when available.
        edge_centers = []
        if envelope is not None:
            edge_xs = [x for x in envelope if x != -1]
            if edge_xs:
                mean_x = float(np.mean(edge_xs))
                edge_centers.append((mean_x, h / 2.0))

        return {
            'stitch_centers': stitch_centers,
            'edge_centers': edge_centers,
            'edge_map': edge_map, # Pass this back if you want to overlay the green line later
            'edge_line_points': edge_line_points,
            'all_distances': all_distances,
            'avg_distance_mm': avg_distance_mm
        }

    def calculate_stitch_edge_distances_vote(self, result):
        """Hybrid selector: prefer YOLO segmentation, fallback to Canny, then last-known.

        Strategy:
            1. Try YOLO segmentation first (robust to lighting / noise).
            2. If YOLO produces 0 distances, fall back to Canny edge detection.
            3. If both fail, return `avg_distance_mm=None` (handled later by fallback logic).
        """

        # Run segmentation first. Only run Canny if segmentation yields no distances.
        seg_res = self.calculate_stitch_edge_distances(result)
        seg_count = len(seg_res.get('all_distances', []))

        if seg_count > 0:
            final_res = seg_res
            vote_source = 'yolo_segmentation'
            canny_res = None
        else:
            canny_res = self.calculate_stitch_edge_distances_canny(result)
            canny_count = len(canny_res.get('all_distances', []))
            if canny_count > 0:
                final_res = canny_res
                vote_source = 'canny'
            else:
                final_res = canny_res
                vote_source = 'none'

        return {
            'stitch_centers': final_res.get('stitch_centers', []),
            'edge_centers': final_res.get('edge_centers', []),
            'edge_map': final_res.get('edge_map'),
            'edge_line_points': final_res.get('edge_line_points'),
            'all_distances': final_res.get('all_distances', []),
            'avg_distance_mm': final_res.get('avg_distance_mm'),
            'vote_source': vote_source,
            'segmentation_result': seg_res,
            'canny_result': canny_res,
        }

    def calculate_measurements(self, predictions, distance_results):
        """
        Calculate stitch measurements from predictions.
        """
        coverage_info = {}
        min_required = getattr(config, "MIN_STITCH_DETECTIONS", 3)
        raw_seam_allowance_mm = distance_results.get('avg_distance_mm')
        if raw_seam_allowance_mm is not None:
            adjusted_seam_allowance_mm = raw_seam_allowance_mm + config.SEAM_ALLOWANCE_OFFSET_MM
            coverage_info["avg_stitch_edge_distance_mm"] = adjusted_seam_allowance_mm
            if not self._is_seam_allowance_in_ideal_range(adjusted_seam_allowance_mm):
                print(
                    "[INFO] seam allowance outside ideal range (override check will decide): "
                    f"{adjusted_seam_allowance_mm:.3f}mm "

                )
        else:
            coverage_info["avg_stitch_edge_distance_mm"] = None

        coverage_info["has_distance_measurement"] = coverage_info["avg_stitch_edge_distance_mm"] is not None

        # Process stitch lengths from predictions
        stitch_lengths = []
        adjusted_stitch_lengths_mm = []
        if predictions is not None and len(predictions) > 0:
            for x1, y1, x2, y2, conf, cls in predictions:
                if int(cls) == config.STITCH_CLASS_ID and conf >= 0.3:
                    width = x2 - x1
                    height = y2 - y1
                    stitch_length_pixels = max(width, height)
                    stitch_length_mm = stitch_length_pixels * self.mm_per_pixel
                    adjusted_stitch_length_mm = stitch_length_mm + config.STITCH_LENGTH_OFFSET_MM
                    adjusted_stitch_lengths_mm.append(adjusted_stitch_length_mm)
                    if not self._is_stitch_length_in_ideal_range(adjusted_stitch_length_mm):
                        print(
                            "[INFO] stitch length outside ideal range (override check will decide): "
                            f"{adjusted_stitch_length_mm:.3f}mm "

                        )

                    stitch_lengths.append({
                        'box': (x1, y1, x2, y2),
                        'length_pixels': stitch_length_pixels,
                        'length_mm': stitch_length_mm,
                        'center': ((x1 + x2) / 2, (y1 + y2) / 2)
                    })

        coverage_info["avg_stitch_length_mm"] = (
            sum(adjusted_stitch_lengths_mm) / len(adjusted_stitch_lengths_mm)
            if adjusted_stitch_lengths_mm else None
        )

        if len(adjusted_stitch_lengths_mm) < min_required:
            coverage_info["avg_stitch_length_mm"] = None

        coverage_info["stitch_lengths"] = stitch_lengths
        return coverage_info

    def process_frame(self, frame, current_total_distance):
        """Process a single frame and return results"""
        if frame is None:
            return None, {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "edge_count": 0,
                "avg_distance_mm": None,
                "avg_stitch_length_mm": None,
                "total_distance_mm": current_total_distance,
                "camera_issue": True,
            }, None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Keep inference silent to avoid per-frame speed/no-detection console spam.
        results = self.model(frame_rgb, device=config.DEVICE, verbose=False)
        result = results[0]

        # Build predictions array: [x1,y1,x2,y2,conf,cls]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()

            valid_indices = confidence >= 0.3
            boxes = boxes[valid_indices]
            classes = classes[valid_indices]
            confidence = confidence[valid_indices]

            preds = np.hstack([boxes, confidence.reshape(-1, 1), classes.reshape(-1, 1)])
        else:
            preds = np.array([])

        # Use both segmentation + Canny and combine their outputs via a vote/weighting system
        dist_res = self.calculate_stitch_edge_distances_vote(result)

        coverage_info = self.calculate_measurements(preds, dist_res)

        # Draw without YOLO segmentation masks or labels.
        annotated = result.plot(masks=False, labels=False)

        # Draw ROI rectangle used for stitch/edge filtering.
        h, w = frame.shape[:2]
        roi_x1 = max(0, min(int(w * EDGE_ROI_LEFT_FRACTION), w - 1))
        roi_x2 = max(roi_x1 + 1, min(int(w * EDGE_ROI_RIGHT_FRACTION), w))
        roi_y1 = max(0, min(int(h * EDGE_ROI_TOP_FRACTION), h - 1))
        roi_y2 = max(roi_y1 + 1, min(int(h * EDGE_ROI_BOTTOM_FRACTION), h))
        cv2.rectangle(
            annotated,
            (roi_x1, roi_y1),
            (roi_x2 - 1, roi_y2 - 1),
            (255, 255, 0),
            2
        )

        seg_res = dist_res.get('segmentation_result') or {}
        seg_contour = seg_res.get('segmentation_contour')
        if seg_contour is not None and len(seg_contour) >= 2:
            contour = seg_contour.astype(np.int32)
            cv2.drawContours(annotated, [contour], -1, (255, 0, 0), 2)

        # Overlay the Canny edge detection result (green) for visualization
        edge_map = dist_res.get('edge_map')
        if edge_map is not None:
            # edge_map is a binary image; overlay it in green
            mask = edge_map > 0
            annotated[mask] = (0, 255, 0)

        edge_line_points = dist_res.get('edge_line_points')
        if edge_line_points and len(edge_line_points) >= 2:
            pts = np.array(edge_line_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

        # Draw seam allowance lines and labels for the selected method
        line_distances = dist_res.get('all_distances', [])
        for dist in line_distances:
            stitch_center = dist.get('stitch_center')
            edge_point = dist.get('edge_point')
            distance_mm = dist.get('distance_mm')
            if stitch_center is None or edge_point is None:
                continue
            cv2.line(
                annotated,
                (int(stitch_center[0]), int(stitch_center[1])),
                (int(edge_point[0]), int(edge_point[1])),
                (0, 255, 0),
                2
            )
            if distance_mm is not None:
                label_mm = distance_mm + config.SEAM_ALLOWANCE_OFFSET_MM
                label = f"SA: {label_mm:.1f}mm"
                label_pos = (int(edge_point[0]) + 5, int(edge_point[1]) - 5)
                cv2.putText(
                    annotated,
                    label,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )

        # Draw stitch centers
        for cx, cy in dist_res['stitch_centers']:
            cv2.circle(annotated, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        # Stitch length labels
        stitch_lengths = coverage_info.get("stitch_lengths", [])
        for stitch in stitch_lengths:
            cx, cy = int(stitch['center'][0]), int(stitch['center'][1])
            length_mm = stitch['length_mm']
            cv2.putText(
                annotated,
                f"{length_mm:.1f}mm",
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1
            )

        stitch_count = len(dist_res['stitch_centers'])
        edge_count = len(dist_res['edge_centers'])

        # Dynamic Y-positioning to avoid overlapping text
        y_pos = 30
        line_spacing = 25  # pixels between lines

        cv2.putText(annotated, f"Total Distance: {current_total_distance:.1f}mm", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        y_pos += line_spacing

        # Avg stitch length
        if coverage_info.get("avg_stitch_length_mm") is not None:
            avg_length = coverage_info["avg_stitch_length_mm"]
            cv2.putText(
                annotated,
                f"Avg Stitch Length: {avg_length:.2f}mm",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y_pos += line_spacing

        # Avg seam allowance (stitch-top edge dist)
        if coverage_info.get("has_distance_measurement") and coverage_info.get("avg_stitch_edge_distance_mm") is not None:
            avg_dist = coverage_info["avg_stitch_edge_distance_mm"]
            cv2.putText(
                annotated,
                f"Avg Seam Allowance: {avg_dist:.2f}mm",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        results_summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "edge_count": edge_count,
            "avg_distance_mm": coverage_info.get("avg_stitch_edge_distance_mm"),
            "avg_stitch_length_mm": coverage_info.get("avg_stitch_length_mm"),
            "total_distance_mm": current_total_distance,
        }

        # ✅ Save latest measurable values for DB thread
        if results_summary["avg_stitch_length_mm"] is not None:
            self.last_avg_stitch_length_mm = results_summary["avg_stitch_length_mm"]
        if results_summary["avg_distance_mm"] is not None:
            self.last_avg_stitch_edge_distance_mm = results_summary["avg_distance_mm"]

        self.last_processed_time = time.time()
        return annotated, results_summary, result