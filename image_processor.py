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

    def _outer_edge_point_for_stitch(self, cx, cy, edge_points, binary_mask=None, y_window=100.0):
        """Return the fabric-side edge point for a stitch.

        If stitches are left of the local edge band, use the right-most edge.
        If stitches are right of it, use the left-most edge. This keeps seam
        allowance lines aimed at the outer contour near the background instead
        of the inner contour nearest to the stitch.
        """
        if edge_points is None:
            return None

        pts = np.asarray(edge_points, dtype=np.float32).reshape(-1, 2)
        if pts.size == 0:
            return None

        local_pts = pts[np.abs(pts[:, 1] - float(cy)) <= float(y_window)]
        if len(local_pts) < 2:
            local_pts = pts

        local_center_x = float(np.median(local_pts[:, 0]))
        use_right_edge = float(cx) <= local_center_x

        min_x = float(np.min(local_pts[:, 0]))
        max_x = float(np.max(local_pts[:, 0]))
        x_span = max_x - min_x
        side_band = max(3.0, min(25.0, x_span * 0.20))

        if use_right_edge:
            side_pts = local_pts[local_pts[:, 0] >= max_x - side_band]
        else:
            side_pts = local_pts[local_pts[:, 0] <= min_x + side_band]

        if len(side_pts) < 2:
            idx = int(np.argmax(local_pts[:, 0]) if use_right_edge else np.argmin(local_pts[:, 0]))
            edge_x, edge_y = float(local_pts[idx, 0]), float(local_pts[idx, 1])
        else:
            [vx, vy, x0, y0] = cv2.fitLine(side_pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
            vx = float(vx[0])
            vy = float(vy[0])
            x0 = float(x0[0])
            y0 = float(y0[0])

            t = (float(cx) - x0) * vx + (float(cy) - y0) * vy
            side_t = (side_pts[:, 0] - x0) * vx + (side_pts[:, 1] - y0) * vy
            if len(side_t) > 0:
                t = float(np.clip(t, np.min(side_t), np.max(side_t)))

            edge_x = x0 + t * vx
            edge_y = y0 + t * vy

        if binary_mask is not None:
            mask_h, _ = binary_mask.shape[:2]
            iy = int(round(edge_y))
            if 0 <= iy < mask_h:
                xs = np.flatnonzero(binary_mask[iy])
                if xs.size > 0:
                    edge_x = float(xs[-1] if use_right_edge else xs[0])
                    edge_y = float(iy)

        return float(edge_x), float(edge_y)

    def _contour_to_polyline(self, contour, closed=False):
        if contour is None or len(contour) < 2:
            return []
        arc_len = cv2.arcLength(contour, closed)
        approx_eps = max(1.0, 0.002 * arc_len)
        approx = cv2.approxPolyDP(contour, approx_eps, closed)
        points = approx.reshape(-1, 2).astype(np.float32)
        polyline = [(float(x), float(y)) for x, y in points]
        if closed and len(polyline) >= 2 and polyline[0] != polyline[-1]:
            polyline.append(polyline[0])
        return polyline

    def _select_fabric_edge_contour(self, contours, stitch_centers, frame_w, frame_h):
        min_length = max(100.0, frame_h * 0.25)
        min_span = max(40, int(frame_h * 0.12))
        candidates = []

        for contour in contours:
            length = cv2.arcLength(contour, False)
            if length < min_length:
                continue

            x, y, cw, ch = cv2.boundingRect(contour)
            if max(cw, ch) < min_span:
                continue

            polyline = self._contour_to_polyline(contour, closed=False)
            if len(polyline) < 2:
                continue

            if stitch_centers:
                distances = []
                for sx, sy in stitch_centers:
                    closest = self._closest_point_on_polyline(float(sx), float(sy), polyline)
                    if closest is None:
                        continue
                    _, _, dist2 = closest
                    distances.append(float(np.sqrt(dist2)))
                if not distances:
                    continue

                median_distance = float(np.median(distances))
            else:
                median_distance = 0.0

            span_bonus = 0.015 * max(cw, ch)
            length_bonus = 0.002 * length
            score = median_distance - span_bonus - length_bonus
            candidates.append((score, contour, polyline))

        if not candidates:
            return None, None

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1], candidates[0][2]

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

        for stitch_center in stitch_centers:
            cx, cy = float(stitch_center[0]), float(stitch_center[1])
            outer_edge = self._outer_edge_point_for_stitch(
                cx,
                cy,
                contour_points,
                binary_mask=combined_edge_mask,
            )
            if outer_edge is None:
                continue

            outer_x, outer_y = outer_edge
            edge_point = (int(round(outer_x)), int(round(outer_y)))
            distance_pixels = float(np.hypot(outer_x - cx, outer_y - cy))
            distance_mm = distance_pixels * self.mm_per_pixel

            all_distances.append({
                'stitch_center': stitch_center,
                'edge_point': edge_point,
                'edge_y': edge_point[1],
                'distance_pixels': distance_pixels,
                'distance_mm': distance_mm
            })

        if len(all_distances) > 0:
            min_dist_mm = min(d['distance_mm'] for d in all_distances)
            thresh = getattr(config, "STITCH_LINE_FILTER_THRESHOLD_MM", 3.5)
            filtered_distances = [d for d in all_distances if d['distance_mm'] <= min_dist_mm + thresh]
            
            valid_distance_count = len(filtered_distances)
            total_distance_mm = sum(d['distance_mm'] for d in filtered_distances)
            avg_distance_mm = total_distance_mm / valid_distance_count if valid_distance_count > 0 else None
            all_distances = filtered_distances
        else:
            avg_distance_mm = None

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
                                smooth_ksize=EDGE_ENVELOPE_SMOOTH_KERNEL,
                                stitch_centers=None):
        """Detect one fabric-edge contour using Canny without left/right bias.

        The returned edge map contains only the selected contour, not every Canny
        edge. This keeps annotation clean and prevents texture/stitch noise from
        becoming the seam allowance target.
        """
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if blur_ksize > 0:
            ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
            gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        edges = cv2.Canny(gray, canny_low, canny_high)

        if dilate_ksize > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ksize, dilate_ksize))
            edges = cv2.dilate(edges, kernel, iterations=1)

        roi_y1 = max(0, min(int(h * roi_top_frac), h - 1))
        roi_y2 = max(roi_y1 + 1, min(int(h * roi_bottom_frac), h))
        roi_x1 = max(0, min(int(w * roi_left_frac), w - 1))
        roi_x2 = max(roi_x1 + 1, min(int(w * roi_right_frac), w))

        mask = np.zeros_like(edges)
        mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255
        edges = cv2.bitwise_and(edges, mask)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        selected_contour, edge_line_points = self._select_fabric_edge_contour(
            contours,
            stitch_centers or [],
            w,
            h,
        )

        selected_edge_map = np.zeros_like(edges)
        if selected_contour is not None:
            cv2.drawContours(selected_edge_map, [selected_contour], -1, 255, 2)

        roi_rect = (roi_x1, roi_y1, roi_x2, roi_y2)
        return selected_edge_map, roi_rect, edge_line_points

    def calculate_stitch_edge_distances_canny(self, result):
        """
        Calculate stitch-to-edge distances using the selected Canny contour.
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

        # 2. Run Canny fallback and keep one selected contour, not a left/right envelope.
        edge_map, _, edge_line_points = self.detect_fabric_edge_canny(
            frame,
            roi_top_frac=roi_y1,
            roi_bottom_frac=roi_y2,
            roi_left_frac=roi_x1,
            roi_right_frac=roi_x2,
            stitch_centers=stitch_centers,
        )

        if edge_line_points is not None and len(edge_line_points) < 2:
            edge_line_points = None

        all_distances = []

        # 3. Calculate perpendicular distances to the selected contour.
        for cx, cy in stitch_centers:
            ix, iy = int(cx), int(cy)

            # Ensure coordinates are within image bounds
            if 0 <= ix < w and 0 <= iy < h and edge_line_points is not None:
                outer_edge = self._outer_edge_point_for_stitch(cx, cy, edge_line_points)
                if outer_edge is None:
                    continue

                edge_x, edge_y = outer_edge
                distance_pixels = float(np.hypot(edge_x - cx, edge_y - cy))
                distance_mm = distance_pixels * self.mm_per_pixel
                edge_point = (int(round(edge_x)), int(round(edge_y)))

                all_distances.append({
                    'stitch_center': (cx, cy),
                    'edge_x': float(edge_x),
                    'edge_point': edge_point,
                    'distance_pixels': float(distance_pixels),
                    'distance_mm': float(distance_mm)
                })

        # 4. Final aggregation with minimum distance filter
        if len(all_distances) > 0:
            min_dist_mm = min(d['distance_mm'] for d in all_distances)
            thresh = getattr(config, "STITCH_LINE_FILTER_THRESHOLD_MM", 3.5)
            filtered_distances = [d for d in all_distances if d['distance_mm'] <= min_dist_mm + thresh]
            
            valid_distance_count = len(filtered_distances)
            total_distance_mm = sum(d['distance_mm'] for d in filtered_distances)
            avg_distance_mm = total_distance_mm / valid_distance_count if valid_distance_count > 0 else None
            all_distances = filtered_distances
        else:
            avg_distance_mm = None

        # If no clean selected contour produced distances, let the caller fall back to buffers.

        # Debug logging: why avg_distance_mm might be None (suppressed - only print if needed for troubleshooting)
        # stitch_count = len(stitch_centers)
        # edge_columns = len(edge_line_points or [])
        # calculated_distances = len(all_distances)
        # print(f"[DEBUG] stitch_centers={stitch_count}, edge_columns={edge_columns}, distances_computed={calculated_distances}, avg_distance_mm={avg_distance_mm}")

        # Provide a minimal edge_centers list so caller code can report an edge count.
        # We use a representative contour point when available.
        edge_centers = []
        if edge_line_points:
            pts = np.array(edge_line_points, dtype=np.float32)
            edge_centers.append((float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))))

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
        raw_adjusted_stitch_lengths_mm = []
        if predictions is not None and len(predictions) > 0:
            for x1, y1, x2, y2, conf, cls in predictions:
                if int(cls) == config.STITCH_CLASS_ID and conf >= 0.3:
                    width = x2 - x1
                    height = y2 - y1
                    stitch_length_pixels = max(width, height)
                    stitch_length_mm = stitch_length_pixels * self.mm_per_pixel
                    adjusted_stitch_length_mm = stitch_length_mm + config.STITCH_LENGTH_OFFSET_MM
                    raw_adjusted_stitch_lengths_mm.append(adjusted_stitch_length_mm)
                    
                    stitch_lengths.append({
                        'box': (x1, y1, x2, y2),
                        'length_pixels': stitch_length_pixels,
                        'length_mm': stitch_length_mm,
                        'center': ((x1 + x2) / 2, (y1 + y2) / 2)
                    })

        # Apply MAD Filter for Stitch Length
        adjusted_stitch_lengths_mm = []
        if len(raw_adjusted_stitch_lengths_mm) > 0:
            median_val = np.median(raw_adjusted_stitch_lengths_mm)
            mad = np.median(np.abs(raw_adjusted_stitch_lengths_mm - median_val))
            # If all values are identical, mad is 0, avoid division by zero
            if mad == 0:
                adjusted_stitch_lengths_mm = raw_adjusted_stitch_lengths_mm
            else:
                for length in raw_adjusted_stitch_lengths_mm:
                    z_score = 0.6745 * abs(length - median_val) / mad
                    if z_score <= getattr(config, "MAD_STITCH_LENGTH_Z_THRESH", 3.5):
                        adjusted_stitch_lengths_mm.append(length)

        for adjusted_stitch_length_mm in adjusted_stitch_lengths_mm:
            if not self._is_stitch_length_in_ideal_range(adjusted_stitch_length_mm):
                print(
                    "[INFO] stitch length outside ideal range (override check will decide): "
                    f"{adjusted_stitch_length_mm:.3f}mm "
                )

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