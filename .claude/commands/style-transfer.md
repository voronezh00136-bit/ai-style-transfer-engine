# Style Transfer Design Skill

Run neural style transfer on images using this project's engine.

## Instructions

When the user invokes this command, help them perform style transfer by:

1. **Check prerequisites**: Verify that `requirements.txt` exists and dependencies are installed. If not, offer to create `requirements.txt` and install dependencies.

2. **Check for transfer.py**: If the main entry point doesn't exist yet, offer to scaffold it with:
   - Argument parsing (--content, --style, --output, --style-weight, --content-weight, --iterations)
   - VGG-19 feature extraction using PyTorch
   - Gram matrix computation for style representation
   - Content and style loss functions
   - L-BFGS or Adam optimization loop
   - Image pre/post-processing with Pillow and torchvision

3. **Run style transfer**: If the engine exists, execute:
   ```
   python transfer.py --content <content_image> --style <style_image> --output <output_path>
   ```
   Ask the user for image paths if not provided as arguments.

4. **Adjust parameters**: Offer to tune:
   - `--style-weight` (default 1e6): Higher = stronger style
   - `--content-weight` (default 1): Higher = preserves more content
   - `--iterations` (default 300): More iterations = finer result
   - `--image-size` (default 512): Output resolution

5. **Review results**: After transfer completes, summarize what was generated and where the output was saved.

## Arguments

$ARGUMENTS - Optional: content and style image paths, e.g. "photo.jpg monet.jpg"
