import os
import shutil

src = "secrets.toml"
dst_dir = ".streamlit"
dst = os.path.join(dst_dir, "secrets.toml")

try:
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
        print(f"Created directory: {dst_dir}")

    if os.path.exists(src):
        # Remove destination if it exists to overwrite
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)
        print(f"Successfully moved {src} to {dst}")
    elif os.path.exists(dst):
        print(f"File already exists at {dst}")
    else:
        print(f"Error: {src} not found in current directory")
except Exception as e:
    print(f"An error occurred: {e}")
