from fastrtc import Stream, ReplyOnPause, get_stt_model
from fastrtc.utils import aggregate_bytes_to_16bit
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.models.s3tokenizer import S3_SR

stt_model = get_stt_model()

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"

tts_model = ChatterboxTTS.from_pretrained(device=DEVICE)

nltk.download('punkt')

llm = init_chat_model("ollama:llama3.2", base_url="http://aurora:11434")

chat_history = [
    SystemMessage(
        "You are AI assistant. Respond in short couple of sentences.")
]

app = FastAPI()


def talk(audio: tuple[int, np.ndarray]):
    print(f"{audio[1][:10]=}")
    prompt = stt_model.stt(audio)
    wav = tts_model.generate(prompt, audio_prompt_path="audio.wav").numpy()

    print(f"{wav[:10]=}")
    yield tts_model.sr, wav


stream = Stream(
    handler=ReplyOnPause(talk),
    modality="audio",
    mode="send-receive"
)

stream.mount(app)


@app.get("/chat")
def chat(message: str) -> str:
    chat_history.append(HumanMessage(message))
    response = llm.invoke(chat_history)
    chat_history.append(response)
    return response.content


@app.get("/")
def _():
    return HTMLResponse(content=open("index.html", "r").read())


def main():
    import uvicorn
    uvicorn.run(app)


if __name__ == "__main__":
    main()
