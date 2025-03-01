import supervision as sv
from inference import get_model
import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def main():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'})
        imageBase64 = data['image']
        img_bytes = base64.b64decode(imageBase64)
        img_array = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image data'})

        model = get_model(model_id="yolov8s-seg-640")

        def callback(image_slice: np.ndarray) -> sv.Detections:
            results = model.infer(image_slice)[0] 
            detections = sv.Detections.from_inference(results)
            return detections

        slicer = sv.InferenceSlicer(callback=callback)
        detections = slicer(image)

        annotated_frame = sv.BoundingBoxAnnotator().annotate(
            scene=image.copy(),
            detections=detections
        )

        # Convert annotated_frame to BGR if it's RGB
        if annotated_frame.shape[2] == 3 and annotated_frame.dtype == np.uint8:
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        else:
            annotated_frame_bgr = annotated_frame

        _, buffer = cv2.imencode('.jpg', annotated_frame_bgr)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        data = {"success": True,"image" :base64_image}
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__== "__main__":
    app.run(debug=True, host='0.0.0.0', port=6060)