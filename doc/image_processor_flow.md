# Image Processor Flow

This document explains the processing flow implemented in image_processor.py and provides detailed flow charts for the main frame pipeline and the stitch-to-edge distance logic.

## Overview

The ImageProcessor class performs these jobs for each camera frame:

- Run YOLO inference to detect stitches and edges.
- Compute stitch length and seam allowance (stitch-to-edge distance).
- Choose the best edge distance source (YOLO segmentation first, Canny fallback).
- Overlay measurements and edge visuals on the output frame.
- Export results for display and for the database thread.

Key data and configuration inputs:

- mm_per_pixel from calibration (pixel to mm conversion).
- Thresholds and offsets from config (min detections, offsets, ideal ranges).
- ROI filtering: only the central 50 percent of the frame is used for stitch/edge centers.

## Full End-to-End Flowchart

```mermaid
flowchart TD
    %% Entry
    A[process_frame(frame, current_total_distance)] --> B{frame is None?}
    B -- Yes --> C[Return (None, summary with camera_issue=True, None)]
    B -- No --> D[Convert BGR to RGB]
    D --> E[Run model inference -> results[0]]
    E --> F[Build predictions array from boxes with conf >= 0.3]
    F --> G[calculate_stitch_edge_distances_vote(result)]
    G --> H[calculate_measurements(preds, dist_res)]
    H --> I[annotated = result.plot(masks=False)]
    I --> J[Overlay edge_map + edge_line_points]
    J --> K[Draw stitch centers + stitch length labels]
    K --> L[Draw text overlays: total distance, avg stitch length, avg seam allowance]
    L --> M[Build results_summary]
    M --> N[Update last_avg_* + last_processed_time]
    N --> O[Return annotated, results_summary, result]

    %% Vote selector
    subgraph Vote[calculate_stitch_edge_distances_vote]
        V1[Run YOLO segmentation distance] --> V2[seg_count]
        V3[Run Canny envelope distance] --> V4[canny_count]
        V2 --> V5{seg_count > 0?}
        V5 -- Yes --> V6[Use segmentation result]
        V5 -- No --> V7{canny_count > 0?}
        V7 -- Yes --> V8[Use Canny result]
        V7 -- No --> V9[Use Canny structure, vote_source = none]
    end
    G --> V1
    G --> V3

    %% YOLO segmentation distance
    subgraph Seg[calculate_stitch_edge_distances]
        S1[Collect stitch centers in central ROI]
        S2[Select highest-confidence edge in ROI]
        S3{stitch_centers < MIN?}
        S4[Return empty distances]
        S5[Build edge mask from best edge]
        S6{edge_centers present?}
        S7[Return avg_distance_mm = None]
        S8{edge mask available?}
        S9[Per stitch: perpendicular top-edge distance]
        S10{any valid distances?}
        S11[avg_distance_mm from mask distances]
        S12[Fallback to top edge y-line]
        S13[avg_distance_mm from y-line distances]
        S14[Build edge_line_points + return]

        S1 --> S2 --> S3
        S3 -- Yes --> S4
        S3 -- No --> S5 --> S6
        S6 -- No --> S7
        S6 -- Yes --> S8
        S8 -- Yes --> S9 --> S10
        S10 -- Yes --> S11 --> S14
        S10 -- No --> S12 --> S13 --> S14
        S8 -- No --> S12
    end
    V6 --> S1

    %% Canny envelope distance
    subgraph Can[calculate_stitch_edge_distances_canny]
        C1[Collect stitch centers in central ROI]
        C2{stitch_centers < MIN?}
        C3[Return empty distances]
        C4[detect_fabric_edge_canny(frame)]
        C5[Compute envelope: bottommost edge per column]
        C6[Per stitch: distance to envelope column]
        C7{any distances?}
        C8[avg_distance_mm from envelope distances]
        C9[Fallback to mean envelope y]
        C10[Return distances + edge_map + edge_line_points]

        C1 --> C2
        C2 -- Yes --> C3
        C2 -- No --> C4 --> C5 --> C6 --> C7
        C7 -- Yes --> C8 --> C10
        C7 -- No --> C9 --> C10
    end
    V8 --> C1
    V9 --> C1

    %% Canny edge detection details
    subgraph Edge[detect_fabric_edge_canny]
        E1[Grayscale + Gaussian blur]
        E2[Canny edge detection]
        E3[Optional dilation]
        E4[Filter contours: keep long edges]
        E5[Apply ROI mask]
        E6[Lower envelope: bottommost edge per column]
        E7[Median smoothing]
        E8[Return envelope, edge_map, roi_rect]
        E1 --> E2 --> E3 --> E4 --> E5 --> E6 --> E7 --> E8
    end
    C4 --> E1

    %% Measurement details
    subgraph Meas[calculate_measurements]
        M1[Seam allowance = avg_distance_mm + SEAM_ALLOWANCE_OFFSET_MM]
        M2[Per stitch: length = max(width, height) * mm_per_pixel]
        M3[Adjusted length = length + STITCH_LENGTH_OFFSET_MM]
        M4{stitch count < MIN?}
        M5[avg_stitch_length_mm = None]
        M6[avg_stitch_length_mm = mean(adjusted lengths)]
        M1 --> M2 --> M3 --> M4
        M4 -- Yes --> M5
        M4 -- No --> M6
    end
    H --> M1
```

## Main Frame Processing Flow

```mermaid
flowchart TD
    A[process_frame(frame, current_total_distance)] --> B{frame is None?}
    B -- Yes --> C[Return (None, summary with camera_issue=True, None)]
    B -- No --> D[Convert BGR to RGB]
    D --> E[Run model inference -> results[0]]
    E --> F[Build predictions array from boxes with conf >= 0.3]
    F --> G[calculate_stitch_edge_distances_vote(result)]
    G --> H[calculate_measurements(preds, dist_res)]
    H --> I[annotated = result.plot(masks=False)]
    I --> J[Overlay edge_map and edge_line_points (green)]
    J --> K[Draw stitch centers and stitch length labels]
    K --> L[Draw text overlays: total distance, avg stitch length, avg seam allowance]
    L --> M[Build results_summary dict]
    M --> N[Update last_avg_* if present; set last_processed_time]
    N --> O[Return annotated, results_summary, result]
```

## Distance Vote (Segmentation vs Canny)

```mermaid
flowchart TD
    A[calculate_stitch_edge_distances_vote(result)] --> B[seg_res = calculate_stitch_edge_distances(result)]
    A --> C[canny_res = calculate_stitch_edge_distances_canny(result)]
    B --> D[seg_count = len(seg_res.all_distances)]
    C --> E[canny_count = len(canny_res.all_distances)]
    D --> F{seg_count > 0?}
    F -- Yes --> G[final = seg_res; vote_source = yolo_segmentation]
    F -- No --> H{canny_count > 0?}
    H -- Yes --> I[final = canny_res; vote_source = canny]
    H -- No --> J[final = canny_res; vote_source = none]
    G --> K[Return final + both method results]
    I --> K
    J --> K
```

## YOLO Segmentation Distance Flow

```mermaid
flowchart TD
    A[calculate_stitch_edge_distances(result)] --> B[Collect stitch centers inside central ROI]
    B --> C[Pick highest-confidence edge detection inside ROI]
    C --> D{stitch_centers < MIN_STITCH_DETECTIONS?}
    D -- Yes --> E[Return empty distances]
    D -- No --> F[Build combined_edge_mask from best edge mask (if available)]
    F --> G{edge_centers present?}
    G -- No --> H[Return with avg_distance_mm = None]
    G -- Yes --> I{edge mask available?}
    I -- Yes --> J[For each stitch: find perpendicular top edge in mask]
    J --> K{any valid distances?}
    K -- Yes --> L[avg_distance_mm from mask distances]
    K -- No --> M[Fallback to top edge y-line from edge centers]
    I -- No --> M
    M --> N[avg_distance_mm from y-line distances]
    L --> O[Build edge_line_points and return]
    N --> O
```

## Canny Envelope Distance Flow

```mermaid
flowchart TD
    A[calculate_stitch_edge_distances_canny(result)] --> B[Collect stitch centers inside central ROI]
    B --> C{stitch_centers < MIN_STITCH_DETECTIONS?}
    C -- Yes --> D[Return empty distances]
    C -- No --> E[detect_fabric_edge_canny(frame)]
    E --> F[Compute envelope: bottommost edge per column]
    F --> G[For each stitch: distance to envelope column]
    G --> H{any distances?}
    H -- Yes --> I[avg_distance_mm from envelope distances]
    H -- No --> J[Fallback to mean envelope y]
    I --> K[Return distances + edge_map + edge_line_points]
    J --> K
```

## Canny Edge Detection Flow

```mermaid
flowchart TD
    A[detect_fabric_edge_canny(frame)] --> B[Convert to grayscale + Gaussian blur]
    B --> C[Canny edge detection]
    C --> D[Optional dilation to bridge gaps]
    D --> E[Find contours and keep long ones]
    E --> F[Apply ROI mask]
    F --> G[Lower envelope: bottommost edge per column]
    G --> H[Median smoothing on envelope]
    H --> I[Return envelope, edge_map, roi_rect]
```

## Measurement Details (Text Summary)

- Seam allowance is derived from avg_distance_mm + config.SEAM_ALLOWANCE_OFFSET_MM.
- Stitch length is computed from each stitch box using max(width, height), then adjusted by config.STITCH_LENGTH_OFFSET_MM.
- If the number of valid stitches is below MIN_STITCH_DETECTIONS, average stitch length is set to None.
- If seam allowance or stitch length is outside the ideal range, the code logs an info message (range checks still rely on a later override decision).
