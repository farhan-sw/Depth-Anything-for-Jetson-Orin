import argparse

import time

import os
from pathlib import Path

import torch
import onnx
import tensorrt as trt
from depth_anything import DepthAnything


def export(
    weights_path: str,  
    save_path: str,
    input_size: int,
    do_onnx: bool = True,
    fp16: bool = False,
):  
    """
    weights_path: str -> Path to the PyTorch model(local / hub)
    save_path: str -> Directory to save the model
    input_size: int -> Width and height of the input image(e.g. 308, 364, 406, 518)
    do_onnx: bool -> Export the model to ONNX format
    fp16: bool -> Use FP16 precision (may cause quality loss on ViT models)
    """
    weights_path = Path(weights_path)
    
    os.makedirs(save_path, exist_ok=True)

    # Load the model
    model = DepthAnything.from_pretrained(weights_path).to('cpu').eval()
    
    # Create a dummy input
    dummy_input = torch.ones((3, input_size, input_size)).unsqueeze(0)
    _ = model(dummy_input)
    onnx_path = Path(save_path) / f"{weights_path.stem}_{input_size}.onnx"
    
    # Export the PyTorch model to ONNX format
    if do_onnx:
        # CRITICAL: dynamo=False forces the legacy TorchScript-based ONNX exporter.
        # torch 2.10 defaults to dynamo=True which produces ONNX graphs that TRT's
        # Myelin backend silently miscompiles (flat/constant output for ViT models).
        # The legacy exporter is battle-tested with TRT and produces correct results.
        torch.onnx.export(
            model,
            dummy_input, 
            onnx_path, 
            opset_version=17, 
            input_names=["input"], 
            output_names=["output"],
            dynamo=False,
        )
        print(f"Model exported to {onnx_path}")
        
        # torch 2.10 may export large models with external data files (.onnx.data)
        # Convert to a single self-contained ONNX file for TensorRT compatibility
        external_data_path = Path(str(onnx_path) + ".data")
        if external_data_path.exists():
            print("Converting external data to single ONNX file...")
            onnx_model = onnx.load(str(onnx_path), load_external_data=True)
            onnx.save_model(
                onnx_model, 
                str(onnx_path),
                save_as_external_data=False,
            )
            # Clean up external data file
            if external_data_path.exists():
                os.remove(external_data_path)
            print("Converted to single ONNX file successfully")
        
        print("ONNX export complete.")
        time.sleep(2)
    
    # ONNX to TensorRT
    onnx_path = onnx_path.resolve()  # Use absolute path
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('Failed to parse the ONNX model.')
    
    print("ONNX model parsed successfully, building TensorRT engine...")
    
    # Set up the builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2 GB
    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 precision")
    else:
        # FP32 — DINOv2 ViT attention layers lose precision in FP16,
        # resulting in flat/gradient-only depth maps without spatial detail.
        print("Using FP32 precision (recommended for DINOv2-based models)")
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    trt_path = onnx_path.with_suffix(".trt")
    with open(trt_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"TensorRT engine saved to {trt_path}")

if __name__ == '__main__':
    # args = argparse.ArgumentParser()
    # args.add_argument("--weights_path", type=str, default="LiheYoung/depth_anything_vits14")
    # args.add_argument("--save_path", type=str, default="weights")
    # args.add_argument("--input_size", type=int, default=406)
    
    # export(
    #     weights_path=args.weights_path,
    #     save_path=args.save_path,
    #     input_size=args.input_size,
    #     onnx=True,
    # )
    
    export(
        weights_path="LiheYoung/depth_anything_vits14", # local hub or online
        save_path="weights", # folder name
        input_size=308, # 308 | 364 | 406 | 518
        do_onnx=False,  # ONNX already exported, skip re-export
        fp16=False,  # FP32 required — DINOv2 ViT loses precision in FP16
    )
