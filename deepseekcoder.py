from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"]
)

# Load model and tokenizer once at startup
tokenizer = AutoTokenizer.from_pretrained("TT5-tee/deepseek-coder", torch_dtype= "auto")
model = AutoModelForCausalLM.from_pretrained("TT5-tee/deepseek-coder")


@app.websocket("/ws/opt")
async def generate_fix_for_opt(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        changes = data.get("changes", [])
        if changes and "optimization" in changes[0]:
            code = changes[0].get("code", "")
            opt = changes[0].get("optimization", "")
            input_text = f"code:{code}\nOptimization:{opt}"
            input_tokens = tokenizer(input_text, return_tensors="pt")
            output = model.generate(**input_tokens, max_new_tokens=512)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            await websocket.send_text(response)


@app.websocket("/ws/fix")
async def generate_fix_for_errors(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        changes = data.get("changes", [])
        if changes and "error" in changes[0]:
            error = changes[0].get("error", "")
            input_text = f"error:{error}\nfix:"
            input_tokens = tokenizer(input_text, return_tensors="pt")
            output = model.generate(**input_tokens, max_new_tokens=512)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            await websocket.send_text(response)
