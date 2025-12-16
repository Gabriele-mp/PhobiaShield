import numpy as np

def compute_iou(box1, box2):
    """Calcola IoU tra due box [cx, cy, w, h]"""
    # Box 1
    b1_x, b1_y, b1_w, b1_h = box1
    x1_min = b1_x - b1_w / 2
    y1_min = b1_y - b1_h / 2
    x1_max = b1_x + b1_w / 2
    y1_max = b1_y + b1_h / 2

    # Box 2
    b2_x, b2_y, b2_w, b2_h = box2
    x2_min = b2_x - b2_w / 2
    y2_min = b2_y - b2_h / 2
    x2_max = b2_x + b2_w / 2
    y2_max = b2_y + b2_h / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = b1_w * b1_h
    box2_area = b2_w * b2_h

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def nms(predictions, iou_threshold=0.10, conf_threshold=None):
    """
    Implementazione fedele di 'nms_cross_class' del compagno.
    """
    if not predictions:
        return []

    # 1. Filtro confidenza opzionale
    if conf_threshold is not None:
        predictions = [p for p in predictions if p['confidence'] >= conf_threshold]

    if not predictions:
        return []

    # --- FASE 1: Standard NMS (per classe) ---
    keep_phase1 = []
    classes = set(p['class_id'] for p in predictions)

    for c in classes:
        class_preds = [p for p in predictions if p['class_id'] == c]
        class_preds.sort(key=lambda x: x['confidence'], reverse=True)

        while class_preds:
            best = class_preds.pop(0)
            keep_phase1.append(best)
            filtered = []
            for p in class_preds:
                if compute_iou(best['bbox'], p['bbox']) < iou_threshold:
                    filtered.append(p)
            class_preds = filtered

    # --- FASE 2: Cross-Class Cleanup (V25 Logic) ---
    # Ordina per AREA (w*h), i più grandi comandano
    keep_phase1.sort(key=lambda x: x['bbox'][2] * x['bbox'][3], reverse=True)
    
    final_keep = []
    
    # Definizioni del compagno
    BIG_BOSS_IDS = [0, 1, 2] # Clown, Shark, Spider
    NOISE_IDS = [3, 4]       # Blood, Needle

    for box in keep_phase1:
        suppressed = False
        for big_box in final_keep:
            iou = compute_iou(big_box['bbox'], box['bbox'])
            
            is_big_boss = big_box['class_id'] in BIG_BOSS_IDS
            is_noise = box['class_id'] in NOISE_IDS
            
            # REGOLA 1: Il Boss mangia il Rumore se si toccano appena (>0.05)
            if is_big_boss and is_noise and iou > 0.05:
                suppressed = True
                break
            
            # REGOLA 2: Se sono classi diverse ma si sovrappongono troppo (>0.2)
            if big_box['class_id'] != box['class_id'] and iou > 0.2:
                suppressed = True
                break
        
        if not suppressed:
            final_keep.append(box)

    return final_keep