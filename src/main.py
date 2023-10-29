from PIL import Image
import requests
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor


image = Image.open("3.jpg")
image.show()

checkpoint = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained('jonastokoliu/image_caption_git-base_pokemon-blip-captions_finetune')


device = "cpu"

inputs = processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values


generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)
