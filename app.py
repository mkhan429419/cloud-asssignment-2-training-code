from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import base64
from io import BytesIO
import os

# Initialize the FastAPI app
app = FastAPI()

# Load the Stable Diffusion model
print("Loading the Stable Diffusion model...")
model_id = "stable-diffusion-model"  # Adjust path if needed
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
).to("cuda")
print("Model loaded successfully!")

# Define the request schema
class PromptRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

@app.get("/")
def root():
    return {"message": "Welcome to the Stable Diffusion API! Use the /generate endpoint to generate images."}

@app.post("/generate")
def generate_image(request: PromptRequest):
    try:
        # Generate the image
        image = pipe(
            request.prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
        ).images[0]

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {"image": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def print_routes():
    for route in app.routes:
        print(f"Path: {route.path}, Methods: {route.methods}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
