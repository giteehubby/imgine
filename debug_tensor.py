#!/usr/bin/env python3

from i2t import jpg2tensor, tensor2pil
from ladder import ladder
import torch
from PIL import Image
import numpy as np

def debug_tensor_shapes():
    """Debug function to understand tensor shape issues"""
    
    # Create a test tensor in the expected format (3, H, W)
    print("=== Testing with synthetic tensor ===")
    test_tensor = torch.randint(0, 256, (3, 100, 100), dtype=torch.uint8)
    print(f'Original tensor shape: {test_tensor.shape}')
    print(f'Original tensor dtype: {test_tensor.dtype}')
    
    # Test ladder function
    result = ladder(test_tensor, 60)
    print(f'After ladder function shape: {result.shape}')
    print(f'After ladder function dtype: {result.dtype}')
    
    try:
        # Try to convert back to PIL
        pil_result = tensor2pil(result)
        print(f'Successfully converted to PIL: {pil_result.size}')
    except Exception as e:
        print(f'Error converting to PIL: {e}')
    
    print("\n=== Testing tensor properties ===")
    print(f'Min value: {result.min()}, Max value: {result.max()}')
    print(f'Unique channels: {result.shape[0] if len(result.shape) > 0 else "N/A"}')
    
    # Check if the tensor has the right number of dimensions
    if len(result.shape) != 3:
        print(f'ERROR: Expected 3D tensor (C, H, W), got {len(result.shape)}D')
    elif result.shape[0] > 4:
        print(f'ERROR: Too many channels: {result.shape[0]} (max should be 4 for RGBA)')

def debug_jpg2tensor():
    """Debug the jpg2tensor function with a sample image"""
    print("\n=== Testing jpg2tensor function ===")
    
    # Create a test image
    test_image = Image.new('RGB', (100, 100), color=(255, 0, 0))
    test_image.save('test_image.jpg')
    
    try:
        # Test the jpg2tensor function
        tensor = jpg2tensor('test_image.jpg')
        print(f'jpg2tensor output shape: {tensor.shape}')
        print(f'jpg2tensor output dtype: {tensor.dtype}')
        
        # Test ladder function on this tensor
        ladder_result = ladder(tensor, 60)
        print(f'After ladder on jpg2tensor shape: {ladder_result.shape}')
        
        # Try to convert back to PIL
        try:
            pil_result = tensor2pil(ladder_result)
            print(f'Successfully converted ladder result to PIL: {pil_result.size}')
        except Exception as e:
            print(f'Error converting ladder result to PIL: {e}')
            
    except Exception as e:
        print(f'Error in jpg2tensor: {e}')
    
    # Test loading the same image with PIL directly
    print("\n=== Testing PIL loading directly ===")
    pil_img = Image.open('test_image.jpg')
    print(f'PIL image mode: {pil_img.mode}')
    print(f'PIL image size: {pil_img.size}')
    
    np_array = np.array(pil_img)
    print(f'NumPy array shape: {np_array.shape}')
    
    # Clean up
    import os
    if os.path.exists('test_image.jpg'):
        os.remove('test_image.jpg')

def debug_jpg2tensor_detailed():
    """Detailed debugging of jpg2tensor to find the 478-channel issue"""
    print("\n=== Detailed jpg2tensor debugging ===")
    
    # Let's examine the jpg2tensor function step by step
    print("Examining jpg2tensor function step by step...")
    
    # Create different types of test images
    test_cases = [
        ('RGB', (100, 100), 'rgb_test.jpg'),
        ('RGBA', (100, 100), 'rgba_test.jpg'),
        ('L', (100, 100), 'grayscale_test.jpg'),
        ('P', (100, 100), 'palette_test.jpg'),
    ]
    
    for mode, size, filename in test_cases:
        try:
            print(f"\n--- Testing {mode} image ---")
            if mode == 'P':
                # Create palette image
                img = Image.new('RGB', size, color=(255, 0, 0))
                img = img.convert('P')
            else:
                img = Image.new(mode, size, color=(255, 0, 0) if mode in ['RGB', 'RGBA'] else 128)
            
            img.save(filename)
            
            # Step-by-step debugging of jpg2tensor
            image = Image.open(filename)
            print(f'Loaded image mode: {image.mode}, size: {image.size}')
            
            image_np = np.array(image)
            print(f'NumPy array shape: {image_np.shape}, dtype: {image_np.dtype}')
            
            # Check the reshape logic
            if image_np.ndim == 3 and image_np.shape[2] == 3:
                print("Applying transpose for RGB image")
                image_np = image_np.transpose((2, 0, 1))
            else:
                print("No transpose applied")
            
            print(f'After potential transpose: {image_np.shape}')
            
            tensor = torch.from_numpy(image_np).to(torch.uint8)
            print(f'Final tensor shape: {tensor.shape}')
            
            # Test ladder on this tensor
            if len(tensor.shape) == 3 and tensor.shape[0] <= 4:
                ladder_result = ladder(tensor, 60)
                print(f'Ladder result shape: {ladder_result.shape}')
                
                try:
                    pil_result = tensor2pil(ladder_result)
                    print(f'PIL conversion successful: {pil_result.size}')
                except Exception as e:
                    print(f'PIL conversion failed: {e}')
            else:
                print(f'Skipping ladder test - invalid tensor shape: {tensor.shape}')
                
        except Exception as e:
            print(f'Error with {mode} image: {e}')
        finally:
            import os
            if os.path.exists(filename):
                os.remove(filename)

if __name__ == "__main__":
    debug_tensor_shapes()
    debug_jpg2tensor()
    debug_jpg2tensor_detailed() 