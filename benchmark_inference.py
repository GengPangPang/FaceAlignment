import time
import torch

from config import (
    IMG_SIZE,
    NUM_LANDMARKS,
    DEVICE,
    EVAL_CHECKPOINT_PATH,
)
from models.HRNet import hrnet_w18_face


try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_dict[k] = v

    model.load_state_dict(new_dict, strict=True)
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_thop_complexity(model, device):
    """
    THOP reports MACs, not always FLOPs.
    Many papers report 1 MAC as 1 operation, while some convert FLOPs = 2 * MACs.
    This function prints both to avoid ambiguity.
    """
    if not HAS_THOP:
        return None

    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        IMG_SIZE,
        IMG_SIZE,
        device=device,
    )

    with torch.no_grad():
        macs, params = profile(
            model,
            inputs=(dummy_input,),
            verbose=False,
        )

    return {
        "macs": macs,
        "flops_2x_macs": 2 * macs,
        "params": params,
    }


def benchmark(
    batch_size=1,
    warmup_iters=50,
    test_iters=300,
    print_complexity=False,
):
    device = torch.device(DEVICE)

    model = hrnet_w18_face(num_landmarks=NUM_LANDMARKS)
    model = load_checkpoint(model, EVAL_CHECKPOINT_PATH, device)
    model = model.to(device)
    model.eval()

    total_params, trainable_params = count_parameters(model)

    complexity = None
    if print_complexity:
        complexity = compute_thop_complexity(model, device)

    dummy_input = torch.randn(
        batch_size,
        3,
        IMG_SIZE,
        IMG_SIZE,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warmup：预热 GPU，避免第一次推理时间异常偏大
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # 正式计时
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(test_iters):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    total_time = end - start
    avg_batch_time = total_time / test_iters
    avg_image_time = avg_batch_time / batch_size
    fps = batch_size / avg_batch_time

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        reserved_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
        gpu_name = torch.cuda.get_device_name(0)
    else:
        peak_memory = None
        reserved_memory = None
        gpu_name = "CPU"

    print("=" * 70)
    print("Inference Benchmark")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"GPU/CPU name: {gpu_name}")
    print(f"Checkpoint: {EVAL_CHECKPOINT_PATH}")
    print(f"Input size: {batch_size} x 3 x {IMG_SIZE} x {IMG_SIZE}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Test iterations: {test_iters}")
    print("-" * 70)
    print(f"Total params by PyTorch: {total_params / 1e6:.3f} M")
    print(f"Trainable params by PyTorch: {trainable_params / 1e6:.3f} M")

    if print_complexity:
        print("-" * 70)
        if complexity is None:
            print("THOP: not installed. Run: pip install thop")
        else:
            print(f"THOP Params: {complexity['params'] / 1e6:.3f} M")
            print(f"MACs: {complexity['macs'] / 1e9:.3f} G")
            print(f"FLOPs, if 1 MAC = 2 FLOPs: {complexity['flops_2x_macs'] / 1e9:.3f} G")

    print("-" * 70)
    print(f"Average batch time: {avg_batch_time * 1000:.3f} ms")
    print(f"Average image time: {avg_image_time * 1000:.3f} ms")
    print(f"FPS: {fps:.2f} images/s")

    if device.type == "cuda":
        print("-" * 70)
        print(f"Peak allocated GPU memory: {peak_memory:.2f} MB")
        print(f"Peak reserved GPU memory: {reserved_memory:.2f} MB")

    print("=" * 70)


if __name__ == "__main__":
    for i, bs in enumerate([1, 8, 16]):
        benchmark(
            batch_size=bs,
            warmup_iters=50,
            test_iters=300,
            print_complexity=(i == 0),  # 只在 batch size=1 时统计一次 THOP
        )