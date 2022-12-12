# CONTAINER ID   NAME                  CPU %     MEM USAGE / LIMIT     MEM %     NET I/O           BLOCK I/O        PIDS
# 0a680c5ef585   checkmailonnxGPU      43.69%    788.1MiB / 30.28GiB   2.54%     66.4MB / 350kB    4.75MB / 147kB   6
# 'scoreDurationMS': 46.34 # using Onnx and Onnx Float.  Onnx Float16 takes about 220 MS

# CONTAINER ID   NAME                  CPU %     MEM USAGE / LIMIT     MEM %     NET I/O           BLOCK I/O       PIDS
# 8ae8b01a58aa   checkmailtensorflow   74.08%    461.8MiB / 30.28GiB   1.49%     13.4MB / 74.9kB   43.1MB / 0B     29
# "scoreDurationMS": 273.37

import numpy as np
from PIL import Image
import json
import time
import io
import hashlib
from datetime import datetime
from http.server import BaseHTTPRequestHandler,HTTPServer
import onnxruntime
import onnx

manifest = json.load(open('cvexport.manifest'))
exportdate = manifest['ExportedDate']
iteration = manifest['IterationId']
model_filepath = manifest['ModelFileName']
lable_filepath = manifest['LabelFileName']
platform = manifest['Platform']
domaintype = manifest['DomainType']
model = None
hardware = None
listentingPort = 80
labels = None
PROB_THRESHOLD = 0.1  # Minimum probably to show results.
providers = [
    'CUDAExecutionProvider',
    'CPUExecutionProvider',
]

with open(lable_filepath, 'r') as f:
    labels = [l.strip() for l in f.readlines()]

def hash_file(filename):
   h = hashlib.sha1()

   with open(filename,'rb') as file:
       chunk = 0
       while chunk != b'':
           chunk = file.read(1024)
           h.update(chunk)

   return h.hexdigest()

class OnnxModel:
    def __init__(self, model_filepath):
        if hash_file(model_filepath) != manifest["ModelFileSHA1"]:
            print(str(datetime.now()) + " ERROR: SHA1 of " + str(model_filepath) + " is " + str(hash_file(model_filepath)) + " should be " + str(manifest["ModelFileSHA1"]))
        else:
            print(str(datetime.now()) + " INFO: SHA1 of " + str(model_filepath) + " verified as " + str(hash_file(model_filepath)))

        self.session = onnxruntime.InferenceSession(str(model_filepath), providers=providers)
        assert len(self.session.get_inputs()) == 1
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[self.session.get_inputs()[0].type]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.is_bgr = False
        self.is_range255 = False
        onnx_model = onnx.load(model_filepath)
        for metadata in onnx_model.metadata_props:
            if metadata.key == 'Image.BitmapPixelFormat' and metadata.value == 'Bgr8':
                self.is_bgr = True
            elif metadata.key == 'Image.NominalPixelRange' and metadata.value == 'NominalRange_0_255':
                self.is_range255 = True

    def predict(self, image):
        image = image.resize(self.input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return json_outputs({name: outputs[i] for i, name in enumerate(self.output_names)})
        
def json_outputs(outputs):
    output = []
    assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
    for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
        if score > PROB_THRESHOLD:
            output.append({'probability': round(float(score), 8),
                        'tagId': int(class_id),
                        'tagName': labels[class_id],
                        'boundingBox': {
                            'left': round(float(box[0]), 8),
                            'top': round(float(box[1]), 8),
                            'width': round(float(box[2]) - float(box[0]), 8),
                            'height': round(float(box[3]) - float(box[1]), 8)
                            }
                        })
    return output

class onnxScore(BaseHTTPRequestHandler):
    def log_request(self, code): 
        pass
    
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(str("Custom Vision Container for edge inferencing.\r\n\r\n").encode("utf-8"))
        self.wfile.write(str("Lables: " + str(labels) + "\r\n").encode("utf-8"))
        self.wfile.write(str("Shape: " + str(model.session.get_inputs()[0].shape[1:]) + "\r\n").encode("utf-8"))
        self.wfile.write(str("Hardware: " + str(hardware) + "\r\n").encode("utf-8"))
        self.wfile.write(str(manifest).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        currentImage = Image.open(io.BytesIO(self.rfile.read(content_length)))

        startTime = time.time()
        predictions = model.predict(currentImage)
        endTime = time.time()
        t_infer = round((endTime-startTime)*1000,2)

        response = {'predictions': predictions,
            'exportedDate': exportdate,
            'currentDateTime': datetime.now().isoformat(),
            'iteration': iteration,
            'scoreDurationMS' : t_infer,
            'labels': labels,
            'platform': platform,
            'domaintype': domaintype,
            'hardware': hardware}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(str(json.dumps(response)).encode("utf-8"))

#        print(response)  
        if len(predictions) > 0:
            print(str(datetime.now()) + " Response: " + str(json.dumps(response)))

def main():
    global model, hardware
    try:
        model = OnnxModel(model_filepath)
        try:
            hardware = onnxruntime.get_device()
        except:
            pass

        server = HTTPServer(('',listentingPort), onnxScore)
        print(str(datetime.now()) + " Server started.  Listneing on port:" + str(listentingPort))
        print(str(datetime.now()) + " Model Shape: " + str(model.session.get_inputs()[0].shape[1:]))
        print(str(datetime.now()) + " Model Type: " + str(model.input_type))
        print(str(datetime.now()) + " Hardware: " + str(hardware))
        print(str(datetime.now()) + " Manifest: " + str(manifest))
        print(str(datetime.now()) + " Labels: " + str(labels))

        server.serve_forever()
    except:
        server.socket.close()   

if __name__ == '__main__':
	main()
