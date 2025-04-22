Note: I intend to revise and update this with more detail in the immediate future.

Description: 

This is a required back-end for my Automatic1111 extension sd-webui-supir_low_vram. It functions by taking API calls from the sd-webui-supir_low_vram extension, upscaling according to the specified parameters and sending the upscaled image back to the extension.
Intended specifically for systems with lower VRAM (12GB), this fork caches models not currently in use to memory, allowing users to upscale images from Automatic1111 during their workflow without the need to worry about VRAM constraints.
The upscaler also functions independent of Auto1111, accessible through a web browser, much like the original SUPIR demo. (Default address: 127.0.0.1:6688). 
All credit for SUPIR goes to https://github.com/Fanghua-Yu/SUPIR and the team mentioned there.

The Automatic1111 extension can be found here: TODO: Update with repo when extension uploaded

Requirements & Installation:
- ~64GB of Memory (RAM + Pagefile) - I've only tested this on a machine using 32GB RAM + 32GB allocated to the pagefile. Most of this is only needed to build the model during startup, but you may need to increase your pagefile if your RAM < 64GB.
  - TODO: Include instructions on increasing pagefile here. (Note: You'll have to Google how to do this for now)
- CUDA - Required for SUPIR. The installation batch file includes torch for CUDA 12.6, if you need to use another CUDA version, you'll need to update this or install requirements seperately. Note that multiple versions of CUDA can be installed simultaneously.
- clip-vit-large-patch14 - Clone this entire repo and place it in the base directory of this repo (the same folder as supir-server.py). The path should look like: "...\SUPIR_Low_VRAM_Backend\clip-vit-large-patch14\..."
- open_clip_pytorch_model.bin - Clone this file and place it in the base directory of this repo. "...\SUPIR_Low_VRAM_Backend\CLIP-ViT-bigG-14-laion2B-39B-b160k\open_clip_pytorch_model.bin"
- At least one of the checkpoints found here: https://drive.google.com/drive/folders/1yELzm5SvAi9e7kPcO_jPp2XkTs4vK6aR
  - SUPIR-v0Q.ckpt produces better results, so you should use that for now, unless you have a specific reason to use SUPIR-v0F.ckpt (or want to experiment)
  - SUPIR-v0F.ckpt is used for denoising which I haven't integrated into this project yet. If you're using this ckpt, you'll need to add the flag "--supir_ckpt_f" in "start_SUPIR.bat", otherwise ckpt_Q will be used as default
  - Place either or both in the QF_CKPTS folder - "\SUPIR_Low_VRAM_Backend\QF_CKPTS"
- An SDXL Checkpoint - I use base SD-XL (https://civitai.com/models/101055/sd-xl), the creators of SUPIR suggest Juggernaut-XL (https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/blob/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors)
  - Place the checkpoint in the SDXL folder - "...\SUPIR_Low_VRAM_Backend\SDXL

All other requirements can be installed by running "install_windows.bat", which creates a venv folder and installs everything else needed.

Running:
- To run, simply run "start_SUPIR.bat"
- The models can take a few minutes to build (up to ~5m depending on your current RAM usage) and it may appear to freeze in the console at a few points. Just give it some time, if you run out of memory, the program will crash rather than freeze.
- When the console outputs the Gradio UI ip address, the models have loaded.
- There is no output directory (yet) for the UI. You'll need to manually save generated images (change the .webp ext to .PNG if you do).
- Images created via Auto1111 are sent back automatically and don't require the UI at all.

Thanks:
- Thank you to the team behind SUPIR
- Thank you to the user behind this reddit post for their instructions on running SUPIR on Low VRAM systems: https://www.reddit.com/r/StableDiffusion/comments/1b37h5z/supir_super_resolution_tutorial_to_run_it_locally/
