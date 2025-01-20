import numpy as np
import cv2
import torch
import logging
from PIL import Image, ImageOps
import mysql.connector
import torch.nn.functional as F
from mmdet.apis import inference_detector
from io import BytesIO
import time
import io
import base64
from tinysam import sam_model_registry, SamPredictor
from mysql.connector import pooling
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

db_pool = None

def initialize_database_pool():
    global db_pool
    db_pool = pooling.MySQLConnectionPool(
        pool_name="mypool",
        pool_size=int(os.getenv('DB_POOL_SIZE', '5')),
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        passwd=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )

# Initialize the database pool when your application starts
initialize_database_pool()

def close_database():
    global global_connection
    if global_connection and global_connection.is_connected():
        global_connection.close()

def get_relevant_classes(unique_id):
    try:
        connection = db_pool.get_connection()
        cursor = connection.cursor()
        sql = """
            SELECT 
                firstEditionHolofoil_market AS firstEditionHolofoil, 
                firstEditionNormal_market AS firstEditionNormal, 
                holofoil_market AS holofoil, 
                normal_market AS normal, 
                reverseHolofoil_market AS reverseHolofoil
            FROM historical_pokemon_card_prices.historical_card_prices
            WHERE unique_id REGEXP %s
            ORDER BY updatedAt DESC
            LIMIT 1
        """
        # Adjust the REGEXP pattern to match your new logic
        # This example assumes 'unique_id' is followed by an underscore and non-digit characters.
        regexp_pattern = '^' + unique_id + '(_[^0-9].*)?$'
        cursor.execute(sql, (regexp_pattern,))
        result = cursor.fetchone()

        if result:
            relevant_classes = [class_name for class_name, value in zip(
                ["firstEditionHolofoil", "firstEditionNormal", "holofoil", "normal", "reverseHolofoil"], 
                result
            ) if value is not None]
            return relevant_classes
        else:
            return []
    except mysql.connector.Error as err:
        print("Database error:", err)
        return []
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def get_bounding_boxes(inference_result, threshold=0.5):
    if hasattr(inference_result, 'pred_instances'):
        pred_instances = inference_result.pred_instances
        bounding_boxes = []
        for bbox, score, label in zip(pred_instances.bboxes, pred_instances.scores, pred_instances.labels):
            if score > threshold:
                bbox_array = bbox.cpu().numpy()
                label = label.item()
                score = score.item()
                bounding_boxes.append((bbox_array, label, score))
        return bounding_boxes
    else:
        raise TypeError("Unexpected result format from the model.")

def segment_objects(encoded_image, rtmdet_model, mask_predictor, DEVICE, threshold=0.5):
    """
    Segment objects from the input image using RTMDet and TinySAM models.
    
    Args:
        encoded_image (str): Base64 encoded image string
        rtmdet_model: RTMDet model for object detection
        mask_predictor: TinySAM model for segmentation
        DEVICE (torch.device): Device to run inference on
        threshold (float): Confidence threshold for detection
        
    Returns:
        tuple: (segmented_images, cropped_objects)
            - segmented_images: List of segmented card images
            - cropped_objects: List of original cropped objects
    """
    logging.debug(f"Segmentation called on device: {DEVICE}")

    start_time = time.time()
    logging.info("Starting segmentation of objects")
    logging.debug(f"segment_objects called with device: {DEVICE}")

    segmented_images = []
    cropped_objects = []

    try:
        image_data = BytesIO(base64.b64decode(encoded_image))
        original_img = ImageOps.exif_transpose(Image.open(image_data))
        original_img_np = np.array(original_img)
        logging.info("Image successfully converted to numpy array")

        rtmdet_model.eval()
        with torch.no_grad():
            inference_result = inference_detector(rtmdet_model, original_img_np)

        bounding_boxes = get_bounding_boxes(inference_result)
        logging.info(f"Found {len(bounding_boxes)} bounding boxes")

        for bbox, label, score in bounding_boxes:
            if score > threshold:
                try:
                    bbox_np = bbox.reshape(-1, 2, 2)
                    mask_predictor.set_image(original_img_np)
                    masks, scores, logits = mask_predictor.predict(box=bbox_np)
                    mask = masks[2]
                    binary_mask = (mask > threshold).astype(np.uint8) * 255
                    mask_area = np.sum(binary_mask) // 255

                    if mask_area > 100:
                        extracted_object = cv2.bitwise_and(original_img_np, original_img_np, mask=binary_mask)
                        y, x = np.where(binary_mask)
                        cropped_object = extracted_object[min(y):max(y), min(x):max(x)]
                        logging.info(f"Cropped object size: {cropped_object.shape}")
                        segmented_images.append(cv2.resize(cropped_object, (299, 299)))
                        cropped_objects.append(cropped_object)
                except Exception as e:
                    logging.error(f"Error processing bounding box: {e}")

        logging.info("Segmentation process completed")
    except Exception as e:
        logging.error(f"Error in segmentation process: {e}")

    return segmented_images, cropped_objects
    logging.info(f"Total object segmentation time: {time.time() - start_time} seconds")


def perform_batch_visual_search(segmented_images, cropped_objects, index, feature_extractor, resnet18_model, class_names, transform, DEVICE, cardGame):
    logging.debug(f"Batch visual search initiated on device: {DEVICE}")

    start_time = time.time()
    logging.debug("Performing batch visual search")

    # Stack and process images in a batch
    batch_imgs = torch.stack([transform(Image.fromarray(img)) for img in segmented_images]).to(DEVICE)

    # Batch feature extraction
    with torch.no_grad():
        batch_features = feature_extractor(batch_imgs)
        batch_embeddings = batch_features.view(batch_features.size(0), -1).cpu().numpy()

    # Query Pinecone and extract ids
    visual_search_results = []

    namespace = cardGame  # Use cardGame as the namespace

    for vector in batch_embeddings:
        query_result = index.query(vector=vector.tolist(), top_k=1, namespace=namespace)
        if query_result.matches:
            visual_search_results.append(query_result.matches[0].id.split('_')[0])
        else:
            visual_search_results.append("No match found")

    # Batch prediction for price types
    price_type_predictions = []
    with torch.no_grad():
        outputs = resnet18_model(batch_imgs)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()

    for i, prob in enumerate(probabilities):
        class_probabilities = dict(zip(class_names, prob))
        relevant_price_types = get_relevant_classes(visual_search_results[i])
        filtered_probs = {k: v for k, v in class_probabilities.items() if k in relevant_price_types}
        price_type_prediction = max(filtered_probs, key=filtered_probs.get, default="none")
        price_type_predictions.append(price_type_prediction)

    # Convert cropped objects to base64
    cropped_image_data = []
    for cropped_img in cropped_objects:
        img_pil_cropped = Image.fromarray(cropped_img)
        buffered = io.BytesIO()
        img_pil_cropped.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        cropped_image_data.append(img_base64)

    return visual_search_results, price_type_predictions, cropped_image_data
    logging.info(f"Total batch visual search processing time: {time.time() - start_time} seconds")

