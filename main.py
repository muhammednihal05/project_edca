from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


from util import load_model, pre_process, post_process 

model = load_model()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    max_age=3600,
)

@app.post("/upload")
async def create_upload_files(file: UploadFile=File(...)):
    image= await file.read()
    
    # Return preprocessed input batch and loaded image
    image = pre_process(image)
    if image=="Null":
        res="neutral"
        return res
    

    # Run the model and postpocess the output
    prediction = model.predict(image)

    # Post process and stitch together the two images to return them
    res = post_process(prediction)
    return res


@app.get("/")
async def main():
 
    return {"message" : "success"}