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
    A[Process frame input] --> B{Frame missing}
    B -- Yes --> C[Return no frame and camera issue summary]
    B -- No --> D[Convert BGR to RGB]
    D --> E[Run model inference]
    E --> F[Build predictions from boxes]
    F --> G[Vote on distance method]
    G --> H[Calculate measurements]
    H --> I[Plot annotations without masks]
    I --> J[Overlay edge map and edge line]
    J --> K[Draw stitch centers and length labels]
    K --> L[Draw text overlays]
    L --> M[Build results summary]
    M --> N[Update last values and time]
    N --> O[Return annotated frame and summary]

    subgraph Vote
        V1[Run segmentation distance]
        V2[Run canny distance]
        V3[Count segmentation distances]
        V4[Count canny distances]
        V5{Seg count positive}
        V6{Canny count positive}
        V7[Use segmentation result]
        V8[Use canny result]
        V9[Use empty result]

        V1 --> V3 --> V5
        V2 --> V4 --> V6
        V5 -- Yes --> V7
        V5 -- No --> V6
        V6 -- Yes --> V8
        V6 -- No --> V9
    end
    G --> V1
    G --> V2

    subgraph Segmentation
        S1[Collect stitch centers in central ROI]
        S2[Select best edge in ROI]
        S3{Stitch count meets minimum}
        S4[Return empty distances]
        S5[Build edge mask]
        S6{Edge centers present}
        S7[Return no average distance]
        S8{Edge mask available}
        S9[Distance per stitch using mask]
        S10{Any valid distances}
        S11[Average distance from mask]
        S12[Fallback to top edge line]
        S13[Average distance from line]
        S14[Build edge line points and return]

        S1 --> S2 --> S3
        S3 -- No --> S4
        S3 -- Yes --> S5 --> S6
        S6 -- No --> S7
        S6 -- Yes --> S8
        S8 -- Yes --> S9 --> S10
        S10 -- Yes --> S11 --> S14
        S10 -- No --> S12 --> S13 --> S14
        S8 -- No --> S12
    end
    V7 --> S1

    subgraph CannyDistance
        C1[Collect stitch centers in central ROI]
        C2{Stitch count meets minimum}
        C3[Return empty distances]
        C4[Detect fabric edge with canny]
        C5[Compute envelope bottommost edge]
        C6[Distance per stitch to envelope]
        C7{Any distances}
        C8[Average distance from envelope]
        C9[Fallback to mean envelope line]
        C10[Return distances and edge data]

        C1 --> C2
        C2 -- No --> C3
        C2 -- Yes --> C4 --> C5 --> C6 --> C7
        C7 -- Yes --> C8 --> C10
        C7 -- No --> C9 --> C10
    end
    V8 --> C1
    V9 --> C1

    subgraph CannyEdge
        E1[Grayscale and blur]
        E2[Canny edge detection]
        E3[Optional dilation]
        E4[Filter long contours]
        E5[Apply ROI mask]
        E6[Lower envelope bottommost edge]
        E7[Median smoothing]
        E8[Return envelope edge map ROI rectangle]
        E1 --> E2 --> E3 --> E4 --> E5 --> E6 --> E7 --> E8
    end
    C4 --> E1

    subgraph Measurements
        M1[Seam allowance from average distance plus offset]
        M2[Per stitch length from max width height]
        M3[Adjust length with offset]
        M4{Stitch count meets minimum}
        M5[Average stitch length none]
        M6[Average stitch length from adjusted values]
        M1 --> M2 --> M3 --> M4
        M4 -- No --> M5
        M4 -- Yes --> M6
    end
    H --> M1
```

## Main Frame Processing Flow

```mermaid
flowchart TD
    A[Process frame input] --> B{Frame missing}
    B -- Yes --> C[Return no frame and camera issue summary]
    B -- No --> D[Convert BGR to RGB]
    D --> E[Run model inference]
    E --> F[Build predictions from boxes]
    F --> G[Vote on distance method]
    G --> H[Calculate measurements]
    H --> I[Plot annotations without masks]
    I --> J[Overlay edge map and edge line]
    J --> K[Draw stitch centers and length labels]
    K --> L[Draw text overlays]
    L --> M[Build results summary]
    M --> N[Update last values and time]
    N --> O[Return annotated frame and summary]
```

## Distance Vote (Segmentation vs Canny)

```mermaid
flowchart TD
    A[Vote on distance method] --> B[Run segmentation distance]
    A --> C[Run canny distance]
    B --> D[Count segmentation distances]
    C --> E[Count canny distances]
    D --> F{Seg count positive}
    F -- Yes --> G[Use segmentation result]
    F -- No --> H{Canny count positive}
    H -- Yes --> I[Use canny result]
    H -- No --> J[Use empty result]
    G --> K[Return selected result]
    I --> K
    J --> K
```

## YOLO Segmentation Distance Flow

```mermaid
flowchart TD
    A[Collect stitch centers in central ROI] --> B[Select best edge in ROI]
    B --> C{Stitch count meets minimum}
    C -- No --> D[Return empty distances]
    C -- Yes --> E[Build edge mask]
    E --> F{Edge centers present}
    F -- No --> G[Return no average distance]
    F -- Yes --> H{Edge mask available}
    H -- Yes --> I[Distance per stitch using mask]
    I --> J{Any valid distances}
    J -- Yes --> K[Average distance from mask]
    J -- No --> L[Fallback to top edge line]
    H -- No --> L
    L --> M[Average distance from line]
    K --> N[Build edge line points and return]
    M --> N
```

## Canny Envelope Distance Flow

```mermaid
flowchart TD
    A[Collect stitch centers in central ROI] --> B{Stitch count meets minimum}
    B -- No --> C[Return empty distances]
    B -- Yes --> D[Detect fabric edge with canny]
    D --> E[Compute envelope bottommost edge]
    E --> F[Distance per stitch to envelope]
    F --> G{Any distances}
    G -- Yes --> H[Average distance from envelope]
    G -- No --> I[Fallback to mean envelope line]
    H --> J[Return distances and edge data]
    I --> J
```

## Canny Edge Detection Flow

```mermaid
flowchart TD
    A[Grayscale and blur] --> B[Canny edge detection]
    B --> C[Optional dilation]
    C --> D[Filter long contours]
    D --> E[Apply ROI mask]
    E --> F[Lower envelope bottommost edge]
    F --> G[Median smoothing]
    G --> H[Return envelope edge map ROI rectangle]
```

## Measurement Details (Text Summary)

- Seam allowance is derived from avg_distance_mm + config.SEAM_ALLOWANCE_OFFSET_MM.
- Stitch length is computed from each stitch box using max(width, height), then adjusted by config.STITCH_LENGTH_OFFSET_MM.
- If the number of valid stitches is below MIN_STITCH_DETECTIONS, average stitch length is set to None.
- If seam allowance or stitch length is outside the ideal range, the code logs an info message (range checks still rely on a later override decision).
