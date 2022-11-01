import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from pydantic import BaseModel
from image_analysis.image_processor import ImageProcessor
from text_analysis.text_processor import TextProcessor

class TextClassifier(nn.Module):
    def __init__(self, num_classes=3,
                 decoder: dict = None):
        super(TextClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(768, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
        self.decoder = decoder

    def forward(self, text):
        x = self.layers(text)
        return x

    def predict(self, text):
        with torch.no_grad():
            x = self.forward(text)
            return x
    
    def predict_proba(self, text):
        with torch.no_grad():
            x = self.forward(text)
            return torch.softmax(x, dim=1)


    def predict_classes(self, text):
        with torch.no_grad():
            x = self.forward(text)
            return self.decoder[int(torch.argmax(x, dim=1))]

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=3,
                 decoder: dict = None):
        super(ImageClassifier, self).__init__()
        device = "cpu"
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = self.resnet50.fc.out_features
        self.linear = nn.Linear(out_features, num_classes).to(device)
        self.layers = nn.Sequential(self.resnet50, self.linear).to(device)
        self.decoder = decoder

    def forward(self, image):
        return self.layers(image)

    def predict(self, image):
        with torch.no_grad():
            return self.forward(image)

    def predict_proba(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return torch.softmax(x, dim=1)

    def predict_classes(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return self.decoder[int(torch.argmax(x, dim=1))]

class TextClassifierCo(nn.Module):
    def __init__(self, num_classes=3):
        super(TextClassifierCo, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(768, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, inp):
        x = self.main(inp)
        return x

class CombinedModel(nn.Module):
    def __init__(self, num_classes=3,
                 decoder: list = None):
        super(CombinedModel, self).__init__()
        device = "cpu"
        resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = resnet50.fc.out_features
        self.image_classifier = nn.Sequential(resnet50, nn.Linear(out_features, 64)).to(device)
        self.text_classifier = TextClassifierCo()
        self.main = nn.Sequential(nn.Linear(128, num_classes))
        self.decoder = decoder

    def forward(self, image_features, text_features):
        image_features = self.image_classifier(image_features)
        text_features = self.text_classifier(text_features)
        combined_features = torch.cat((image_features, text_features), 1)
        combined_features = self.main(combined_features)
        return combined_features

    def predict(self, image_features, text_features):
        with torch.no_grad():
            combined_features = self.forward(image_features, text_features)
            return combined_features
    
    def predict_proba(self, image_features, text_features):
        with torch.no_grad():
            x = self.forward(image_features, text_features)
            return torch.softmax(x, dim=1)

    def predict_classes(self, image_features, text_features):
        with torch.no_grad():
            x = self.forward(image_features, text_features)
            return self.decoder[int(torch.argmax(x, dim=1))]

class TextItem(BaseModel):
    text: str

try:
    text_processor = TextProcessor(max_length=20)
    with open('/Users/behzad/AiCore/Facebook_Marketplace_RRS/ml_models/text_decoder.pkl', 'rb') as f:
        text_decoder = pickle.load(f)

    n_classes = len(text_decoder)
    txt_classifier = TextClassifier(num_classes=n_classes, decoder=text_decoder)
    txt_classifier.load_state_dict(torch.load('/Users/behzad/AiCore/Facebook_Marketplace_RRS/ml_models/model_bert.pt', map_location='cpu'))
except:
    raise OSError("No Text model found. Check that you have the decoder and the model in the correct location")

try:
    image_processor = ImageProcessor()
    with open('/Users/behzad/AiCore/Facebook_Marketplace_RRS/ml_models/image_decoder.pkl', 'rb') as f:
        image_decoder = pickle.load(f)

    n_classes = len(image_decoder)
    img_classifier = ImageClassifier(num_classes=n_classes, decoder=image_decoder)
    img_classifier.load_state_dict(torch.load('/Users/behzad/AiCore/Facebook_Marketplace_RRS/ml_models/resnet50.pt', map_location='cpu'))
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

try:
    text_processor = TextProcessor(max_length=20)
    image_processor = ImageProcessor()
    with open('/Users/behzad/AiCore/Facebook_Marketplace_RRS/ml_models/combined_decoder.pkl', 'rb') as f:
        combined_decoder = pickle.load(f)

    n_classes = len(combined_decoder)
    combined = CombinedModel(num_classes=n_classes, decoder=combined_decoder)
    combined.load_state_dict(torch.load('/Users/behzad/AiCore/Facebook_Marketplace_RRS/ml_models/combined_model.pt', map_location='cpu'))
except:
    raise OSError("No Combined model found. Check that you have the encoder and the model in the correct location")

try:
    text_processor = TextProcessor(max_length=20)
except:
    raise OSError("No Text processor found. Check that you have the encoder and the model in the correct location")

try:
    image_processor = ImageProcessor()
except:
    raise OSError("No Image processor found. Check that you have the encoder and the model in the correct location")

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

@app.post('/predict/text')
def predict_text(text: str = Form(...)):
    
    processed_txt = text_processor(text)
    prediction = txt_classifier.predict(processed_txt)
    probs = txt_classifier.predict_proba(processed_txt)
    classes = txt_classifier.predict_classes(processed_txt)
    print(prediction)
    print(probs)
    print(classes)
    return JSONResponse(status_code=200, content={'prediction': prediction.tolist(), 'probs': probs.tolist(), 'classes': classes})
  
  
@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):

    img = Image.open(image.file)
    processed_img = image_processor(img)
    prediction = img_classifier.predict(processed_img)
    probs = img_classifier.predict_proba(processed_img)
    classes = img_classifier.predict_classes(processed_img)
    print(prediction)
    print(probs)
    print(classes)
    return JSONResponse(status_code=200, content={'prediction': prediction.tolist(), 'probs': probs.tolist(), 'classes': classes})
  
@app.post('/predict/combined')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    print(text)
    processed_txt = text_processor(text)
    img = Image.open(image.file)
    processed_img = image_processor(img)
    prediction = combined.predict(processed_img, processed_txt)
    probs = combined.predict_proba(processed_img, processed_txt)
    classes = combined.predict_classes(processed_img, processed_txt)
    print(prediction)
    print(probs)
    print(classes)
    return JSONResponse(status_code=200, content={'prediction': prediction.tolist(), 'probs': probs.tolist(), 'classes': classes})
    
if __name__ == '__main__':
    uvicorn.run("api:app", host="0.0.0.0", port=8080)