import sys
import os
from PIL import Image

def process_and_embed_image(input_path, output_h_path, output_png_path=None, target_size=(224, 224)):
    """
    Resizes an image, converts it to RGB, and generates a C header file
    containing its raw bytes as a static array.
    Optionally saves the processed image as a PNG file.
    """
    print(f"Processing '{input_path}'...")

    # 1. Process the image (Resize and convert to RGB)
    try:
        img = Image.open(input_path)
        img = img.convert('RGB')  # Ensure standard RGB color space
        # Use LANCZOS for high-quality downscaling
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Save the processed image as PNG if output path is provided
        if output_png_path:
            img.save(output_png_path, format='PNG')
            print(f"Saved processed image to '{output_png_path}'.")

        # Save to a temporary buffer to get the raw PNG bytes
        import io
        byte_io = io.BytesIO()
        img.save(byte_io, format='PNG')
        img_bytes = byte_io.getvalue()

        print(f"Image resized to {target_size} and converted to PNG format.")

    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        return
    except Exception as e:
        print(f"Error during image processing: {e}")
        return

    # 2. Generate the C header file for static allocation
    print(f"Generating C header file '{output_h_path}'...")
    try:
        with open(output_h_path, 'w') as f:
            variable_name = "bear_statue_image"
            
            f.write(f"#ifndef {variable_name.upper()}_H\n")
            f.write(f"#define {variable_name.upper()}_H\n\n")
            f.write(f"// Statically allocated image data from '{input_path}'\n")
            f.write(f"// Processed to {target_size} RGB PNG\n")
            f.write(f"#include <stddef.h>\n\n")

            f.write(f"static const size_t {variable_name}_len = {len(img_bytes)};\n\n")
            f.write(f"// Placed in read-only memory section (.rodata)\n")
            f.write(f"static const unsigned char {variable_name}_data[] = {{\n    ")

            for i, byte in enumerate(img_bytes):
                f.write(f"0x{byte:02X}")
                # Add a comma unless it's the very last byte
                if i < len(img_bytes) - 1:
                    f.write(", ")
                # Add a newline every 16 bytes for readability
                if (i + 1) % 16 == 0:
                    f.write("\n    ")

            f.write(f"\n}};\n\n")
            f.write(f"#endif // {variable_name.upper()}_H\n")

        print(f"Successfully generated '{output_h_path}'.")
        print(f"Total static memory used: {len(img_bytes)} bytes.")

    except Exception as e:
        print(f"Error writing header file: {e}")

if __name__ == "__main__":
    # Input file from the user prompt
    input_image = '/scratch2/agustin/merlin/third_party/iree_bar/samples/simple_embedding_glpdepth/UCB_bear.jpg'
    
    # Output header file name
    output_header = 'bear_image_data.h'
    
    # Output PNG file name
    output_png = 'processed_image.png'
    
    process_and_embed_image(input_image, output_header, output_png)