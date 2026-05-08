uv run generate-new.py --size 480*832 \
      --ckpt_dir ./models --convert_model_dtype \
      --image examples/i2v_input.JPG \
      --sample_steps 8 --sample_guide_scale 1.0 \
      --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
