import os

from sup_logging import logger
import gradio as gr
import argparse
from SUPIR.util import HWC3, upscale_image, fix_resize, convert_dtype
import numpy as np
import torch
from SUPIR.util import create_SUPIR_model, load_Q_or_F_ckpt, check_QF_dir
from PIL import Image
import einops
import copy
import time
import random
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--opt", type=str, default='options/SUPIR_v0.yaml')
parser.add_argument("--ip", type=str, default='127.0.0.1')
parser.add_argument("--port", type=int, default='6688')
parser.add_argument("--no_llava", action='store_true', default=False)
parser.add_argument("--log_history", action='store_true', default=False)
parser.add_argument("--loading_half_params", action='store_true', default=False)
parser.add_argument("--use_tile_vae", action='store_true', default=False)
parser.add_argument("--encoder_tile_size", type=int, default=512)
parser.add_argument("--decoder_tile_size", type=int, default=64)
parser.add_argument("--load_8bit_llava", action='store_true', default=False)
parser.add_argument("--no_param_reporting", action='store_true', default=False)
parser.add_argument("--supir_ckpt_f", action='store_true', default=False)
args = parser.parse_args()
server_ip = args.ip
server_port = args.port
param_reporting: bool = not args.no_param_reporting
if args.supir_ckpt_f:
    QF_sign = 'F'   # Fidelity
else:
    QF_sign = 'Q'   # Quality

# Check for QF_CKPTs pre-model loading to avoid lengthy wait time before crashing

if not check_QF_dir(args.opt, QF_sign):
    logger.error(f"Checkpoint file 'SUPIR-v0{QF_sign}.ckpt' could not be found. Ensure the file exists at the above "
                 f"directory and that its location is specified in '{args.opt}'.")
    if QF_sign == 'F':
        if check_QF_dir(args.opt, 'Q'):
            logger.warning(f"SUPIR-v0Q.ckpt was found. To use SUPIR with ckpt_Q, relaunch without '--supir_ckpt_f' flag.")
        else:
            logger.error(f"Neither SUPIR-v0F.ckpt or SUPIR-v0Q.ckpt could be found.")
    elif QF_sign == 'Q':
        if check_QF_dir(args.opt, 'F'):
            logger.warning(f"SUPIR-v0F.ckpt was found. To use SUPIR with ckpt_F, relaunch with '--supir_ckpt_f' flag.")
        else:
            logger.error(f"Neither SUPIR-v0Q.ckpt or SUPIR-v0F.ckpt could be found.")
    quit()

if torch.cuda.device_count() >= 2:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:1'
elif torch.cuda.device_count() == 1:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')


def create_SUPIR(sign='Q'):
    model, default_setting = create_SUPIR_model(args.opt, SUPIR_sign=sign, load_default_setting=True)
    if args.loading_half_params:
        model = model.half()
    if args.use_tile_vae:
        model.init_tile_vae(encoder_tile_size=args.encoder_tile_size, decoder_tile_size=args.decoder_tile_size)
    model = model.to(SUPIR_device)
    # model.first_stage_model.denoise_encoder_s1 = copy.deepcopy(model.first_stage_model.denoise_encoder)
    model.current_model = f'v0-{sign}'
    # ckpt_Q, ckpt_F = load_QF_ckpt(args.opt)
    ckpt_QF = load_Q_or_F_ckpt(args.opt, SUPIR_sign=sign)

    """
    if sign == 'F':
        ckpt_QF = load_F_ckpt(args.opt)
    else:
        ckpt_QF = load_Q_ckpt(args.opt)
    """

    return ckpt_QF, model, default_setting


def return_default_txt_prompts():
    a_prompt = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - ' \
               'realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore ' \
               'detailing, hyper sharpness, perfect without deformations.'
    n_prompt = 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, ' \
               '3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, ' \
               'signature, jpeg artifacts, deformed, lowres, over-smooth'
    return a_prompt, n_prompt


def call_upscaler(input_image, upscale=2, seed=-1, steps=50, prompt_cfg=1.0, s2_str=1.0, noise=1.003,
                  a_prompt=return_default_txt_prompts()[0], n_prompt=return_default_txt_prompts()[1]):

    print('')
    model.to(SUPIR_device)
    gc.collect()
    #torch.cuda.empty_cache()

    prompt = ''
    num_samples = 1
    edm_steps = int(min(200, max(steps, 1)))
    s_stage1 = -1.0     # Stage 1 = -1.0 as we're not using llava
    s_stage2 = min(1.0, max(s2_str, 0.0))
    s_cfg = min(15.0, max(prompt_cfg, 1.0))
    s_churn = 5
    s_noise = min(1.1, max(noise, 1.0))
    color_fix_type = 'Wavelet'
    diff_dtype = 'fp16'
    ae_dtype = 'bf16'
    gamma_correction = 1.0
    linear_CFG = True
    linear_s_stage2 = False
    spt_linear_CFG = default_setting.spt_linear_CFG_Quality
    spt_linear_s_stage2 = 0.0
    model_select = model.current_model

    if param_reporting:
        logger.info(f"UPSCALING PARAMETERS: \nUpscale Amount = {upscale} \nSeed = {seed} \nSteps = {edm_steps}"
                    f"\nCFG - {s_cfg} \nStage 2 Strength = {s_stage2} \nNoise = {s_noise} \n \nPositive Prompt: {a_prompt}"
                    f"\n \nNegative Prompt: {n_prompt} \n")

    if seed == -1:
        seed = random.randint(0, 65535)
        if param_reporting:
            logger.info(f"Random Seed Generated: {seed} \n")

    results = stage2_process(input_image, prompt, a_prompt, n_prompt, num_samples, upscale, edm_steps, s_stage1, s_stage2,
                   s_cfg, seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype, gamma_correction, linear_CFG,
                   linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select)

    result: Image.Image = results[0]

    print('')
    print("*" * 80)
    if param_reporting:
        logger.info(f"UPSCALE COMPLETE -> Upscale: ({upscale})x, Steps: {edm_steps}, CFG: {s_cfg}, Noise: {s_noise}")
    else:
        logger.info(f"UPSCALE COMPLETE")
    print("*" * 80)

    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    return result


def stage2_process(input_image, prompt, a_prompt, n_prompt, num_samples, upscale, edm_steps, s_stage1, s_stage2,
                   s_cfg, seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype, gamma_correction,
                   linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select):
    torch.cuda.set_device(SUPIR_device)
    event_id = str(time.time_ns())
    event_dict = {'event_id': event_id, 'localtime': time.ctime(), 'prompt': prompt, 'a_prompt': a_prompt,
                  'n_prompt': n_prompt, 'num_samples': num_samples, 'upscale': upscale, 'edm_steps': edm_steps,
                  's_stage1': s_stage1, 's_stage2': s_stage2, 's_cfg': s_cfg, 'seed': seed, 's_churn': s_churn,
                  's_noise': s_noise, 'color_fix_type': color_fix_type, 'diff_dtype': diff_dtype, 'ae_dtype': ae_dtype,
                  'gamma_correction': gamma_correction, 'linear_CFG': linear_CFG, 'linear_s_stage2': linear_s_stage2,
                  'spt_linear_CFG': spt_linear_CFG, 'spt_linear_s_stage2': spt_linear_s_stage2,
                  'model_select': model_select}

    if model_select != model.current_model:
        if model_select == 'v0-Q':
            print('load v0-Q')
            model.load_state_dict(ckpt_Q, strict=False)
            model.current_model = 'v0-Q'
        elif model_select == 'v0-F':
            print('load v0-F')
            model.load_state_dict(ckpt_F, strict=False)
            model.current_model = 'v0-F'
    input_image = HWC3(input_image)
    input_image = upscale_image(input_image, upscale, unit_resolution=32,
                                min_size=1024)

    LQ = np.array(input_image) / 255.0
    LQ = np.power(LQ, gamma_correction)
    LQ *= 255.0
    LQ = LQ.round().clip(0, 255).astype(np.uint8)
    LQ = LQ / 255 * 2 - 1
    LQ = torch.tensor(LQ, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
    captions = ['']

    model.ae_dtype = convert_dtype(ae_dtype)
    model.model.dtype = convert_dtype(diff_dtype)

    samples = model.batchify_sample(LQ, captions, num_steps=edm_steps, restoration_scale=s_stage1, s_churn=s_churn,
                                    s_noise=s_noise, cfg_scale=s_cfg, control_scale=s_stage2, seed=seed,
                                    num_samples=num_samples, p_p=a_prompt, n_p=n_prompt, color_fix_type=color_fix_type,
                                    use_linear_CFG=linear_CFG, use_linear_control_scale=linear_s_stage2,
                                    cfg_scale_start=spt_linear_CFG, control_scale_start=spt_linear_s_stage2)

    x_samples = (einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(
        0, 255).astype(np.uint8)
    results = [x_samples[i] for i in range(num_samples)]

    if args.log_history:
        os.makedirs(f'./history/{event_id[:5]}/{event_id[5:]}', exist_ok=True)
        with open(f'./history/{event_id[:5]}/{event_id[5:]}/logs.txt', 'w') as f:
            f.write(str(event_dict))
        f.close()
        Image.fromarray(input_image).save(f'./history/{event_id[:5]}/{event_id[5:]}/LQ.png')
        for i, result in enumerate(results):
            Image.fromarray(result).save(f'./history/{event_id[:5]}/{event_id[5:]}/HQ_{i}.png')
    return results


def use_default_params():
    upscale_amount = 1
    pos_prompt, neg_prompt = return_default_txt_prompts()
    steps = 50
    prompt_cfg = 7.5
    ctrl_str = 1.0
    noise = 1.003
    return upscale_amount, pos_prompt, neg_prompt, steps, prompt_cfg, ctrl_str, noise


try:
    ckpt_QF, model, default_setting = create_SUPIR(QF_sign)
    model.cpu()
    gc.collect()
    # torch.cuda.empty_cache()
    print('')
    logger.info(f"Successfully created SUPIR model (Ckpt_{QF_sign}). Starting Gradio... \n")
except Exception as e:
    logger.error(f"Error creating SUPIR model: {e}")
    ckpt_QF = "Failed_To_Load_CKPT"
    quit()


try:
    block = gr.Blocks(title='SUPIR', analytics_enabled=False).queue()
    with block:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", elem_id="image-input")
                reset_params_button = gr.Button(value="Reset Params To Default", size='md')
                upscale_amount = gr.Slider(label="Upscale Amount", minimum=1, maximum=8, value=1, step=1, elem_id="upscale_amount")
                seed = gr.Slider(label="Seed", info="Set to -1 to generate random seed", minimum=-1, maximum=2147483647,
                                 step=1, randomize=True)
                pos_prompt = gr.Textbox(label="Prompt", value=return_default_txt_prompts()[0])
                neg_prompt = gr.Textbox(label="Negative Prompt", value=return_default_txt_prompts()[1])
                steps = gr.Slider(label="Steps", minimum=1, maximum=200, value=50, step=1)
                prompt_cfg = gr.Slider(label="Prompt CFG", minimum=1.0, maximum=15.0, value=7.5, step=0.1)
                ctrl_str = gr.Slider(label="Guidance Strength", minimum=0., maximum=1., value=1., step=0.05)
                noise = gr.Slider(label="Noise", minimum=1.0, maximum=1.1, value=1.003, step=0.001)
            with gr.Column():
                output_image = gr.Image(type="pil", elem_id="image-output")
                do_stage2_button = gr.Button(value="Generate")
                event_id = gr.Textbox(label="Event ID", value="", visible=False)

        reset_params_button.click(fn=use_default_params, outputs=[upscale_amount, pos_prompt, neg_prompt, steps,
                                  prompt_cfg, ctrl_str, noise])
        do_stage2_button.click(fn=call_upscaler, inputs=[input_image, upscale_amount, seed, steps, prompt_cfg,
                                                         ctrl_str, noise, pos_prompt, neg_prompt], outputs=output_image)
    block.launch(server_name=server_ip, server_port=server_port)
except Exception as e:
    logger.error(f"Error in gradio server: {e}")
