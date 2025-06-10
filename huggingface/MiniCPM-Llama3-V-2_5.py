# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True)