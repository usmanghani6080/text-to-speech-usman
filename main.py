from fastapi import FastAPI, UploadFile, File
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import torch
import numpy as np

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

@app.post("/generate_audio/")
async def generate_audio(description: str, prompt: str):
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    return audio_arr.tolist(), model.config.sampling_rate

@app.post("/convert_to_faster_audio/")
async def convert_to_faster_audio(description: str, prompt: str):
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    # Increase the playback speed
    faster_audio_arr = np.interp(np.arange(0, len(audio_arr), 1.5), np.arange(0, len(audio_arr)), audio_arr).astype(np.int16)

    return faster_audio_arr.tolist(), model.config.sampling_rate

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
