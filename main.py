from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import base64
import uvicorn
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch 

is_cuda_available = torch.cuda.is_available()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# load the transformer model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
if is_cuda_available:
    model = model.to('cuda')

def read_file_as_image(data) :
    img = Image.open(io.BytesIO(data)).convert('RGB')
    return img

@app.get("/ping")
async def ping() :
    # Just for debugging purpose
    return "Server is alive"

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate-caption")
async def predict(request: Request, file: UploadFile = File(...)) :
    resp_file = None
    resp = None
    form = await request.form()
    n = form.get("n")
    max_length = form.get("max_length")
    pre_condition = form.get("pre_condition").strip()
    img_type = form.get("file").content_type
    parsed_n_m = False
    try:
        n = int(n)
        max_length = int(max_length)
        parsed_n_m = True
    except Exception as e:
        resp = {"request": request, "error_message": f"'Number of Captions' and 'Max Length of Caption' must be an integer."}
        resp_file = 'error.html'
    if parsed_n_m and (n < 1 or max_length < 1):
        resp = {"request": request, "error_message": "'Number of Captions' and 'Max Length of Caption' must be greater than 0."}
        resp_file = 'error.html'
    else:
        try:
            raw_image_file_content = await file.read()
            encoded_image = base64.b64encode(raw_image_file_content).decode("utf-8")
            raw_image = read_file_as_image(raw_image_file_content)
            if len(pre_condition) > 0:
                inputs = processor(raw_image, pre_condition, return_tensors="pt")
            else:
                inputs = processor(raw_image, return_tensors="pt")
            if is_cuda_available:
                inputs = inputs.to('cuda')
            out = model.generate(
                num_return_sequences=n, 
                do_sample=True, 
                max_length=max_length,
                temperature=0.8, 
                top_k=n+3, 
                top_p=0.95, 
                **inputs)
            decoded_outputs = processor.batch_decode(out, skip_special_tokens=True)
            resp = {"request": request, 
                    "n": n, 
                    "img": encoded_image,
                    "img_type": img_type,
                    "captions": decoded_outputs}
            resp_file = 'captions.html'
        except Exception as e:
            resp = {"request": request, "error_message": f"An exception occurred while processing file. Exception: {e}"}
            resp_file = 'error.html'
    return templates.TemplateResponse(resp_file, resp)

if __name__ == "__main__" :
    uvicorn.run(app, host = 'localhost', port = 8000)