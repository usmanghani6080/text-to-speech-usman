from fastapi import FastAPI, File, HTTPException
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import torch
import soundfile as sf
import tempfile
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")


class SynthesizeRequest(BaseModel):
    prompt: str
    description: str


@app.post("/generate")
async def synthesize_text(request: SynthesizeRequest):
    try:
        input_ids = tokenizer(request.description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids.to(device)

        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()

        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_filename = tmp_file.name
            sf.write(tmp_filename, audio_arr, model.config.sampling_rate)

        return FileResponse(tmp_filename, media_type='audio/wav')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def read_root():
    return {"message": "Welcome to Parler TTS synthesis API. Please use the /synthesize endpoint."}

