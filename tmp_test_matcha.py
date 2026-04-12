import torch
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image

def main():
    model_id = "google/matcha-chartqa"
    print("Loading model...")
    processor = Pix2StructProcessor.from_pretrained(model_id)
    # Load on CPU for a quick test
    model = Pix2StructForConditionalGeneration.from_pretrained(model_id)

    # Let's test on one image from the dataset, e.g., Readings_2023-12-09.png
    img_path = "/nas/longleaf/home/txueying/CGM-VQA/graphs_cropped/Readings_2023-12-09.png"
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print("Image not found: ", e)
        return

    # Questions to test
    questions = [
        # Yes/no question
        ("Did the largest meal event lead to a glucose spike?", "Did the largest meal event lead to a glucose spike? (yes/no)"),
        ("Did the largest meal event lead to a glucose spike?", "Is the largest meal event leading to a glucose spike? Answer yes or no."),
        # Temporal question 
        ("When did the largest glucose spike occur?", "When did the largest glucose spike occur?"),
        ("Locate the segment where glucose declined after an insulin event.", "What time did glucose decline after insulin?")
    ]

    for q_orig, q_test in questions:
        print(f"\n--- Testing Prompt ---")
        print(f"Prompt: {q_test}")
        inputs = processor(images=image, text=q_test, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32)
        
        ans = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Answer: {ans}")

if __name__ == "__main__":
    main()
