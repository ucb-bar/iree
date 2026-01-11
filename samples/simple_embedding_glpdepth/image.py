import numpy as np
from PIL import Image

def generate_header(input_image, output_header="bear_image_data.h"):
    # 1. Load and Resize
    img = Image.open(input_image).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    
    # 2. Convert to Model Format: [1, 3, 224, 224] Float32, 0.0-1.0
    # Numpy defaults to HWC, we need CHW for PyTorch/ONNX models
    data = np.array(img).astype(np.float32) / 255.0
    data = data.transpose(2, 0, 1) # HWC -> CHW
    
    # Flatten to raw bytes
    raw_bytes = data.tobytes()

    # 3. Write C Header
    print(f"Generating {output_header} with {len(raw_bytes)} bytes of raw float data...")
    with open(output_header, 'w') as f:
        f.write(f"#ifndef BEAR_IMAGE_DATA_H\n#define BEAR_IMAGE_DATA_H\n\n")
        f.write(f"#include <stddef.h>\n\n")
        f.write(f"// Raw float32 pixel data: [3, 224, 224] (CHW), Normalized 0-1\n")
        f.write(f"static const size_t bear_statue_image_len = {len(raw_bytes)};\n")
        f.write(f"static const unsigned char bear_statue_image_data[] __attribute__((aligned(16))) = {{\n")
        
        # Write bytes in hex
        for i in range(0, len(raw_bytes), 16):
            chunk = raw_bytes[i:i+16]
            hex_str = ", ".join(f"0x{b:02X}" for b in chunk)
            f.write(f"    {hex_str},\n")
            
        f.write(f"}};\n\n#endif\n")

if __name__ == "__main__":
    generate_header("/scratch2/agustin/merlin/third_party/iree_bar/samples/simple_embedding_glpdepth/UCB_bear.jpg") # Replace with your image filename