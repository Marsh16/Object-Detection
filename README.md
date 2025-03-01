# Small Object Detection API (Flask, YOLO, Small Model, Base64 Image Input)

This API provides a robust and efficient solution for detecting small objects in images, leveraging a Flask web framework and a pre-trained YOLO (You Only Look Once) model optimized for small object detection. It is designed to receive image data encoded in base64 format via HTTP POST requests, making it ideal for applications where direct file uploads are not feasible or desired.

## Key Features

* **Flask-based REST API:** Offers a straightforward and accessible RESTful interface for image processing.
* **YOLO for Small Object Detection:** Employs a specialized, lightweight YOLO model variant, finely tuned for detecting tiny objects, balancing speed and accuracy. This ensures optimal performance in resource-constrained environments or applications with strict latency requirements.
* **Base64 Image Input:** Accepts image data encoded as base64 strings within the JSON payload of POST requests. This facilitates seamless integration with various client-side applications and data transmission methods.
* **JSON Output:** Returns detection results in a structured JSON format, including bounding box coordinates, class labels, and confidence scores for each detected object.
* **HTTP POST Method:** Utilizes the POST method for secure and reliable data transmission of the base64 encoded image.
* **Efficient Image Processing:** Decodes the base64 string, processes the image using the YOLO model, and returns the detection results rapidly.
* **Scalability:** Flask's architecture allows for potential scaling to handle increased request volumes.
* **Easy Integration:** The standardized JSON output and base64 input simplify integration with diverse applications and platforms.

## Use Cases

* Web-based applications requiring image analysis without direct file uploads.
* Mobile applications transmitting image data in base64 format.
* Remote monitoring systems sending image snapshots as base64 strings.
* Applications where network bandwidth optimization is crucial.
* Integration with systems that only support text-based data transfer.
* Any situation where small objects must be found, and base64 image data is prefered.

## Technical Details

* Implemented using Python and the Flask web framework.
* Integrates a pre-trained YOLO model (e.g., custom-trained or a lightweight version of YOLOv5 or YOLOv8) optimized for small object detection.
* Utilizes libraries such as Pillow (PIL) and NumPy for efficient image processing.
* Accepts base64 encoded image data within the JSON payload of POST requests.
* Returns detection results in JSON format, including bounding box coordinates, class labels, and confidence scores.

## How to Use

1.  Encode the image as a base64 string.
2.  Send a POST request to the API endpoint with a JSON payload containing the base64 encoded image.
3.  The API decodes the base64 string, processes the image using the YOLO model, and returns a JSON response containing the detection results.

## Example (Python Client)

```python
import requests
import base64
import json

def predict_image(image_path, api_url):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    payload = {"image": encoded_string}
    headers = {'Content-type': 'application/json'}
    response = requests.post(api_url, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Request failed with status code {response.status_code}"}

image_path = "path/to/your/image.jpg" #replace with your image path
api_url = "[http://127.0.0.1:6060/predict](http://127.0.0.1:6060/predict)" #local api url, change if needed
results = predict_image(image_path, api_url)
print(json.dumps(results, indent=4))