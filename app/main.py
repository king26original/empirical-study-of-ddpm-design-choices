from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import torch
import io
from PIL import Image
import os
from huggingface_hub import hf_hub_download
from src.model.single_head_attention import model
from src.model.multi_head_attention import model_multi
from src.evaluation.multi_image_sampler import sample_batch
from src.diffusion.cosine_beta_schedule import cosine_alpha, cosine_alpha_bar, cosine_posterior_variance
from src.diffusion.linear_beta_schedule import linear_alpha, linear_alpha_bar, linear_posterior_variance

HF_REPO_ID = "king11010/ddpm"

app = FastAPI(
    title="DDPM Inference API", 
    description="REST API for generating images using the trained Denoising Diffusion Probabilistic Model",
    version="1.0.0"
)

# Global variables to hold the model in memory
model1=None
model2=None
ema_model3=None
ema_model4=None

def tensor_to_pil(tensor):
    tensor=tensor.detach().cpu()
    # tensor shape: [1, 3, 32, 32]
    img = tensor[0] # remove batch dim -> [3, 32, 32]
    # Denormalize from [-1, 1] to [0, 255]
    img = ((img + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    # Change shape to [32, 32, 3] for PIL
    img = img.permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def download_models():
    """
    Download model weights and vocab from HuggingFace Hub.
    hf_hub_download caches files locally after the first download,
    so this is fast on subsequent calls.

    Returns:
        Tuple of (encoder_path, decoder_path) as strings
    """
    model1_path= hf_hub_download(repo_id=HF_REPO_ID, filename="model1")
    model2_path= hf_hub_download(repo_id=HF_REPO_ID, filename="model2")
    ema_model3_path= hf_hub_download(repo_id=HF_REPO_ID, filename="ema_model3")
    ema_model4_path= hf_hub_download(repo_id=HF_REPO_ID, filename="ema_model4")
    return model1_path, model2_path, ema_model3_path, ema_model4_path

@app.on_event("startup")
async def load_model():
    """Loads the model into memory when the server starts."""
    global model1, model2, ema_model3, ema_model4
    
    try:
        model1=model()
        model2=model()
        ema_model3=model()
        ema_model4=model_multi()

        model1_path, model2_path, ema_model3_path, ema_model4_path=download_models()

        model1.load_state_dict(torch.load(model1_path, map_location=torch.device('cpu')))
        model2.load_state_dict(torch.load(model2_path, map_location=torch.device('cpu')))
        ema_model3.load_state_dict(torch.load(ema_model3_path, map_location=torch.device('cpu')))
        ema_model4.load_state_dict(torch.load(ema_model4_path, map_location=torch.device('cpu')))

        model1.eval()
        model2.eval()
        ema_model3.eval()
        ema_model4.eval()
        
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

@app.get("/health")
def health_check():
    """SDE Best Practice: Always include a health check endpoint."""
    return {
        "status": "healthy", 
        "model1_loaded": model1 is not None,
        "model2_loaded": model2 is not None,
        "ema_model3_loaded": ema_model3 is not None,
        "ema_model4_loaded": ema_model4 is not None,
        "service": "DDPM Inference API"
    }

@app.post("/generate")
async def generate_image():
    """Generates images and returns them as a PNG stream."""
    
    if model1 is None:
        raise HTTPException(status_code=503, detail="Model1 is not loaded. Please check server logs.")
    if model2 is None:
        raise HTTPException(status_code=503, detail="Model2 is not loaded. Please check server logs.")
    if ema_model3 is None:
        raise HTTPException(status_code=503, detail="Model3 is not loaded. Please check server logs.")
    if ema_model4 is None:
        raise HTTPException(status_code=503, detail="Model4 is not loaded. Please check server logs.")

    try:

        image1=sample_batch(model1, linear_alpha, linear_alpha_bar,linear_posterior_variance,batch_size=1)
        image2=sample_batch(model2, cosine_alpha, cosine_alpha_bar, cosine_posterior_variance, batch_size=1)
        image3=sample_batch(ema_model3, cosine_alpha, cosine_alpha_bar, cosine_posterior_variance, batch_size=1)
        image4=sample_batch(ema_model4, cosine_alpha, cosine_alpha_bar, cosine_posterior_variance, batch_size=1)

        # # --- Dummy image for testing the API structure ---
        # image = Image.new('RGB', (32, 32), color='red') 
        # # -------------------------------------------------

        # Convert PIL Image to bytes for streaming response
        img_byte_arr = io.BytesIO()

        image1=tensor_to_pil(image1)
        image2=tensor_to_pil(image2)
        image3=tensor_to_pil(image3)
        image4=tensor_to_pil(image4)

        grid = Image.new('RGB', (64, 64))
        grid.paste(image1, (0, 0))
        grid.paste(image2, (32, 0))
        grid.paste(image3, (0, 32))
        grid.paste(image4, (32, 32))

        grid.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")