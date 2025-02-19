from huggingface_hub import HfApi
from huggingface_hub import whoami

access_token = "hf_****"

api = HfApi(token=access_token)

api.upload_folder(
    folder_path="/mnt/petrelfs/zhaoshitian/data/ai2d/mvr-ai2d",
    repo_id="stzhao/MVR-AI2D",
    path_in_repo="data/",
    repo_type="dataset"
    # allow_patterns="*.txt", # Upload all local text files
    # delete_patterns="*.txt", # Delete all remote text files before
)