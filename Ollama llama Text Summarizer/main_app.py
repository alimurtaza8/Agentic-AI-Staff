from PIL import Image
import pytesseract
import os
from ollama_llm import llm_response

# install pytesseract for interacting with the images
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def convert_images_to_text(image_file):
    try:
        image = Image.open(image_file)
        # Extract text from the image using pytesseract
        extracted_text = pytesseract.image_to_string(image)

        # Save the extracted text to a text file
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        save_path = f"{image_name}_extracted_text.txt"

        # Write the text which is extracted from the image to back the new file
        with open(save_path, 'w') as file:
            file.write(extracted_text)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def llm_summary(file_path):
  with open(file_path, 'r') as file:
    prompt = f" Write a Clear summary of this text {file.read()}"
    print(llm_response(prompt))

image_path = "/file.jpg"  #Replace your image file here.
convert_images_to_text(image_path)

text_file_path = "file_extracted_text.txt" 
llm_summary(text_file_path)