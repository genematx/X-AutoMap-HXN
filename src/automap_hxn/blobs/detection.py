import numpy as np
import cv2
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border
from ..utils import normalize_and_dilate

# Cellpose imports (optional - will gracefully handle if not installed)
try:
    from cellpose import models
    from PIL import Image
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    models = None
    Image = None

# Cache for Cellpose models to avoid reloading on every detection call
# Key: (model_type, gpu), Value: CellposeModel instance
_CELLPOSE_MODEL_CACHE = {}


def detect_blobs(img_norm, img_orig, min_thresh, min_area, color, 
                 file_name, method='simple', 
                 include_method_info=False, **kwargs):
    """
    General blob detection function that supports multiple detection methods.
    
    Parameters:
    -----------
    img_norm : np.ndarray
        Normalized image for detection
    img_orig : np.ndarray  
        Original image for intensity calculations
    min_thresh : float
        Minimum threshold for detection
    min_area : float
        Minimum area for blob filtering
    color : str
        Color label for the blobs
    file_name : str
        Name of the file being processed
    method : str
        Detection method to use. Options:
        - 'simple': OpenCV SimpleBlobDetector (default) - Good for general circular/elliptical blobs
        - 'contours': Contour-based detection - Good for irregular shapes
        - 'hough': Hough circle detection - Best for perfect circles
        - 'connected_components': Connected components labeling - Fast, good for well-separated objects
        - 'watershed': Watershed segmentation - Good for touching/overlapping objects
        - 'cellpose': Cellpose deep learning segmentation - Best for cells and complex biological objects
    include_method_info : bool
        If True, includes 'method' key in output for compatibility (default: False)
    **kwargs : dict
        Additional method-specific parameters:
        
        For 'simple' method:
            max_threshold=255, max_area=1600, threshold_step=2,
            filter_by_color=False, filter_by_circularity=False, etc.
            
        For 'hough' method:
            max_radius=40, dp=1, min_dist=20, param1=50, param2=30
            
        For 'watershed' method:
            min_distance=10, threshold_abs=0.3
            
        For 'cellpose' method:
            diameter=60, model_type='cyto3', gpu=False, flow_threshold=0.4,
            cellprob_threshold=0.0, channels=[0,0], min_diameter=0, max_diameter=inf
        
    Returns:
    --------
    list : List of detected blob dictionaries with keys:
        'Box', 'center', 'radius', 'color', 'file', 
        'max_intensity', 'mean_intensity', 'mean_dilation',
        'box_x', 'box_y', 'box_size'
        (plus 'method' key if include_method_info=True)
        
    Examples:
    ---------
    # Basic usage (default simple method) - SAME OUTPUT FORMAT AS BEFORE
    blobs = detect_blobs(img_norm, img_orig, 50, 100, 'red', 'test.tiff')
    
    # Use contour detection for irregular shapes
    blobs = detect_blobs(img_norm, img_orig, 50, 100, 'red', 'test.tiff', method='contours')
    
    # Use Hough circles with custom parameters 
    blobs = detect_blobs(img_norm, img_orig, 50, 100, 'red', 'test.tiff', 
                        method='hough', max_radius=50, min_dist=30)
                        
    # Use Cellpose for biological samples
    blobs = detect_blobs(img_norm, img_orig, 50, 100, 'red', 'test.tiff',
                        method='cellpose', diameter=60, model_type='cyto3')
                        method='contours', include_method_info=True)
                        
    # Compare multiple methods (automatically includes method info)
    results = detect_blobs_multi_method(img_norm, img_orig, 50, 100, 'red', 'test.tiff',
                                       methods=['simple', 'contours', 'hough'])
    """
    
    # Method dispatch
    method_map = {
        'simple': _detect_blobs_simple,
        'contours': _detect_blobs_contours, 
        'hough': _detect_blobs_hough_circles,
        'connected_components': _detect_blobs_connected_components,
        'watershed': _detect_blobs_watershed,
        'cellpose': _detect_blobs_cellpose
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown detection method: {method}. Available: {list(method_map.keys())}")
    
    # Special check for Cellpose availability
    if method == 'cellpose' and not CELLPOSE_AVAILABLE:
        raise ImportError(f"Cellpose not available. Install with: pip install cellpose[gui]")
    
    # Apply morphological preprocessing (normalize and dilate)
    # EXCEPTION: Skip for cellpose - deep learning models need raw/original images
    # Morphological dilation can destroy fine details that cellpose was trained to recognize
    if method == 'cellpose': #TODO not clean fix later
        # Use original images for cellpose (no morphological preprocessing)
        processed_norm = img_norm
        processed_dilated = img_orig
    else:
        # Apply morphological preprocessing for all other methods
        processed_norm, processed_dilated = normalize_and_dilate(img_orig, 
                                                                 kernel_size=3, 
                                                                 iterations=1)
    
    # Detect blobs using the selected method
    detections = method_map[method](processed_dilated, processed_norm, min_thresh, min_area, **kwargs)
    
    # Convert detections to standard format
    blobs = []
    for idx, detection in enumerate(detections, start=1):
        x, y = detection['center']
        radius = detection['radius']
        box_size = 2 * radius
        box_x, box_y = x - radius, y - radius

        x1, y1 = max(0, box_x), max(0, box_y)
        x2, y2 = min(processed_norm.shape[1], x + radius), min(processed_norm.shape[0], y + radius)
        roi_orig = processed_norm[y1:y2, x1:x2]
        roi_dilated = processed_dilated[y1:y2, x1:x2]

        if roi_orig.size > 0:
            blob_dict = {
                'Box': f"{file_name} Box #{idx}",
                'center': (x, y),
                'radius': radius,
                'color': color,
                'file': file_name,
                'max_intensity': roi_orig.max(),
                'mean_intensity': roi_orig.mean(),
                'mean_dilation': float(roi_dilated.mean()),
                'box_x': box_x,
                'box_y': box_y,
                'box_size': box_size
            }
            
            # Only add method info if requested for backward compatibility
            if include_method_info:
                blob_dict['method'] = method
                
            blobs.append(blob_dict)
    
    return blobs


def _detect_blobs_cellpose(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Cellpose-based blob detection for cell/particle segmentation"""
    if not CELLPOSE_AVAILABLE:
        raise ImportError("Cellpose not available. Install with: pip install cellpose")
    
    # Use img_orig (normalized but NOT dilated) because Cellpose is a deep learning model
    # trained on raw images. Morphological dilation can destroy fine details.
    # img_norm = dilated image (used for simple/contour methods)
    # img_orig = normalized but not dilated (better for deep learning models)
    cellpose_input = img_orig
    
    # Convert to format expected by Cellpose
    if len(cellpose_input.shape) == 2:
        # Convert grayscale to RGB format for Cellpose
        img_rgb = np.stack([cellpose_input, cellpose_input, cellpose_input], axis=2)
    else:
        img_rgb = cellpose_input.copy()
    
    # Normalize to [0,1] range
    img_min, img_max = float(img_rgb.min()), float(img_rgb.max())
    if img_max > img_min:
        img_rgb = (img_rgb - img_min) / (img_max - img_min)
    else:
        # Handle constant image
        return []
    
    # Cellpose parameters
    diameter_guess = kwargs.get('diameter', 30)
    model_type = kwargs.get('model_type', 'cyto3')
    gpu = kwargs.get('gpu', False)
    flow_threshold = kwargs.get('flow_threshold', 0.4)
    cellprob_threshold = kwargs.get('cellprob_threshold', 0.0)
    channels = kwargs.get('channels', [0, 0])  # [cytoplasm, nucleus] channels
    print(f"Running Cellpose with  '{model_type = }' "
          f"and {diameter_guess = }..., "
          f"{gpu = }")
    
    # Initialize model (with caching to avoid reloading)
    cache_key = (model_type, gpu)
    if cache_key not in _CELLPOSE_MODEL_CACHE:
        print(f"Loading Cellpose model: {model_type} (GPU={gpu})...")
        _CELLPOSE_MODEL_CACHE[cache_key] = models.CellposeModel(pretrained_model=model_type, gpu=gpu)
        print(f"Cellpose model loaded and cached.")
    else:
        print(f"Using cached Cellpose model: {model_type} (GPU={gpu})")
    
    model = _CELLPOSE_MODEL_CACHE[cache_key]
    
    # Run detection
    try:
        # Use min_size from kwargs if provided, otherwise fall back to min_area
        cellpose_min_size = kwargs.get('min_size', min_area)
        
        res = model.eval(
            img_rgb,
            channels=channels,
            diameter=diameter_guess,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=cellpose_min_size
        )

        
        # res = model.eval(
        #     img_rgb,
        #     channels=[0,0],
        #     diameter=30,
        #     flow_threshold=0.4,
        #     cellprob_threshold=0)
        
        # Handle different return formats
        if len(res) == 4:
            masks, flows, styles, diams = res
        else:
            masks, flows, styles = res
        import matplotlib.pyplot as plt
        plt.imshow(masks)
        plt.show()
            
    except Exception as e:
        print(f"Cellpose detection failed: {e}")
        return []
    masks = clear_border(masks) #to clear edge boxes
    # Convert masks to boxes and areas
    boxes, areas = _masks_to_boxes_and_areas(masks)
    
    # Filter by diameter range if specified
    min_diameter = kwargs.get('min_diameter', 0)
    max_diameter = kwargs.get('max_diameter', float('inf'))
    
    detections = []
    for box, area in zip(boxes, areas):
        # Check area threshold
        if area < min_area:
            continue
            
        # Check diameter threshold
        equiv_diameter = _area_to_equiv_diameter(area)
        if not (min_diameter <= equiv_diameter <= max_diameter):
            continue
        
        # Calculate center and radius from bounding box
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Use equivalent radius from area for consistency
        radius = equiv_diameter / 2
        
        detections.append({
            'center': (int(center_x), int(center_y)),
            'radius': int(radius),
            'area': area,
            'equiv_diameter': equiv_diameter,
            'bbox': box
        })
    
    return detections


def _detect_blobs_watershed(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Watershed segmentation for blob detection"""
    from scipy import ndimage
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    
    # Apply threshold
    _, binary = cv2.threshold(img_norm, min_thresh, 255, cv2.THRESH_BINARY)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # Find local maxima as markers
    local_max_coords = peak_local_max(
        dist_transform, 
        min_distance=kwargs.get('min_distance', 10),
        threshold_abs=kwargs.get('threshold_abs', 0.3 * dist_transform.max())
    )
    
    # Create markers
    markers = np.zeros_like(binary, dtype=np.int32)
    for i, (y, x) in enumerate(local_max_coords):
        markers[y, x] = i + 1
    
    # Apply watershed
    labels = watershed(-dist_transform, markers, mask=binary)
    
    detections = []
    for label_id in np.unique(labels):
        if label_id == 0:  # Skip background
            continue
        
        mask = labels == label_id
        area = np.sum(mask)
        
        if area >= min_area:
            # Calculate centroid
            y_coords, x_coords = np.where(mask)
            x = int(np.mean(x_coords))
            y = int(np.mean(y_coords))
            radius = int(np.sqrt(area / np.pi))
            detections.append({'center': (x, y), 'radius': radius})
    
    return detections


# Helper functions for convenient method-specific detection

def _masks_to_boxes_and_areas(masks):
    """
    Convert Cellpose masks to bounding boxes and areas.
    
    Returns:
        boxes: list of (x1, y1, x2, y2)
        areas: list of mask pixel areas (same order as boxes)
    """
    boxes, areas = [], []
    ids = np.unique(masks)
    ids = ids[ids != 0]  # Skip background
    
    for i in ids:
        ys, xs = np.where(masks == i)
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        boxes.append((x1, y1, x2, y2))
        areas.append(int(xs.size))
        
    return boxes, areas

def detect_blobs_simple(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):
    """Convenient wrapper for simple blob detection"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name, 
                       method='simple', include_method_info=include_method_info, **kwargs)

def detect_blobs_contours(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):
    """Convenient wrapper for contour-based blob detection"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                       method='contours', include_method_info=include_method_info, **kwargs)

def detect_blobs_hough(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):  
    """Convenient wrapper for Hough circle detection"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                       method='hough', include_method_info=include_method_info, **kwargs)

def detect_blobs_connected_components(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):
    """Convenient wrapper for connected components detection"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                       method='connected_components', include_method_info=include_method_info, **kwargs)

def detect_blobs_watershed(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):
    """Convenient wrapper for watershed segmentation detection"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                       method='watershed', include_method_info=include_method_info, **kwargs)

def detect_blobs_cellpose(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):
    """Convenient wrapper for Cellpose deep learning segmentation"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                       method='cellpose', include_method_info=include_method_info, **kwargs)


def get_available_detection_methods():
    """Returns list of available detection methods"""
    methods = ['simple', 'contours', 'hough', 'connected_components', 'watershed']
    if CELLPOSE_AVAILABLE:
        methods.append('cellpose')
    return methods


def detect_blobs_multi_method(img_norm, img_orig, min_thresh, min_area, color, file_name, 
                             methods=['simple'], combine_results=True, **kwargs):
    """
    Apply multiple detection methods and optionally combine results.
    
    Parameters:
    -----------
    methods : list
        List of detection methods to apply
    combine_results : bool  
        If True, combine all results into single list. If False, return dict by method.
    **kwargs : dict
        Additional parameters for detection methods
        
    Returns:
    --------
    list or dict : Combined results or dict of results by method
    """
    all_results = {}
    
    for method in methods:
        try:
            blobs = detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                               method=method, include_method_info=True, **kwargs)
            all_results[method] = blobs
            print(f"Method '{method}': Found {len(blobs)} blobs")
        except Exception as e:
            print(f"Error with method '{method}': {e}")
            all_results[method] = []
    
    if combine_results:
        # Combine all results (method info already included via include_method_info=True)
        combined_blobs = []
        for method, blobs in all_results.items():
            combined_blobs.extend(blobs)
        return combined_blobs
    
    return all_results


# ---------------- File wrapper ----------------
def _detect_blobs_simple(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Simple blob detector method (OpenCV SimpleBlobDetector)"""
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = min_thresh
    params.maxThreshold = kwargs.get('max_threshold', 255)
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = kwargs.get('max_area', 1600)
    params.thresholdStep = kwargs.get('threshold_step', 2)

    params.filterByColor = kwargs.get('filter_by_color', False)
    params.filterByCircularity = kwargs.get('filter_by_circularity', False)
    params.filterByInertia = kwargs.get('filter_by_inertia', False)
    params.filterByConvexity = kwargs.get('filter_by_convexity', False)
    params.minRepeatability = kwargs.get('min_repeatability', 1)
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img_norm)
    
    detections = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = int(kp.size / 2)
        detections.append({'center': (x, y), 'radius': radius})
    
    return detections

def _detect_blobs_contours(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Contour-based blob detection"""
    # Apply threshold
    _, binary = cv2.threshold(img_norm, min_thresh, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            detections.append({'center': (int(x), int(y)), 'radius': int(radius)})
    
    return detections

def _detect_blobs_hough_circles(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Hough circle detection for circular blobs"""
    # Convert min_area to min_radius (assuming circular blobs)
    min_radius = int(np.sqrt(min_area / np.pi))
    max_radius = kwargs.get('max_radius', 40)
    
    circles = cv2.HoughCircles(
        img_norm,
        cv2.HOUGH_GRADIENT,
        dp=kwargs.get('dp', 1),
        minDist=kwargs.get('min_dist', min_radius * 2),
        param1=kwargs.get('param1', 50),
        param2=kwargs.get('param2', 30),
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    detections = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            detections.append({'center': (x, y), 'radius': r})
    
    return detections

def _detect_blobs_connected_components(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Connected components labeling for blob detection"""
    # Apply threshold
    _, binary = cv2.threshold(img_norm, min_thresh, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    detections = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x, y = int(centroids[i][0]), int(centroids[i][1])
            # Estimate radius from area
            radius = int(np.sqrt(area / np.pi))
            detections.append({'center': (x, y), 'radius': radius})
    
    return detections

def _area_to_equiv_diameter(area_px):
    """Convert area to equivalent circle diameter: A = π (d/2)^2  -> d = 2*sqrt(A/π)"""
    return 2.0 * np.sqrt(area_px / np.pi)
