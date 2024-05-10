import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)
import moviepy.editor as mp

import glob 
from tqdm import tqdm
paths = glob.glob("BagOfLies/Finalised/User_*/run_*/")
for path in tqdm(paths[315:]):
    path = path + "video.mp4"
    clip = mp.VideoFileClip(path)
    path = path.replace("video.mp4", "audio.wav")
    clip.audio.write_audiofile(path)
    result = pipe(path)
    path = path.replace("audio.wav", "transcript.txt")
    with open(path, "w") as file:
        file.write(result["text"])