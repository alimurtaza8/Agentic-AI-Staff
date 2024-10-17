# AI Text Summarizer

## Objective
The objective of this project is to create an AI-driven text summarizer using the **Ollama Llama 3.1** model. The summarizer extracts text from images using **Pytesseract** and generates concise summaries by interacting with the **Llama 3.1** language model.

## 1. Text Extraction and Summarization
The project operates in two main stages:

- **Text Extraction:** We use `Pytesseract` to extract text from images.
- **Text Summarization:** The extracted text is then summarized using the **Llama 3.1** model, generating clear, concise summaries.

## 2. Data Processing and Preparation
The flow of data processing includes:

- **Image to Text Conversion:** Uploaded images are processed using `Pytesseract` for OCR, converting visual data into textual information.
- **Summarization:** The extracted text is passed to the **Llama 3.1** model, which generates an accurate and meaningful summary based on the input.

## 3. How to Use the Project
This project is designed to run in **Google Colab**. To set up and run the project, follow these steps:
**NOTE DO NOT RUN THIS PROJECT IN LOCAL MACHINE**.


### Step 1: Set Up Environment in Google Colab

1. Install **Colab-Xterm** to open a terminal:
   ```bash
   !pip install colab-xterm
   %load_ext colabxterm
   %xterm
   ```

2. In the terminal, install **Ollama** and set up the **Llama 3.1** model:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama serve & ollama run llama3.1
   ```

### Step 2: Install Tesseract for OCR
To convert images into text, you'll need **Tesseract**:

```bash
!apt-get install tesseract-ocr
!apt-get install libtesseract-dev
!pip install pytesseract
```

### Step 3: Run the Text Summarizer

1. Upload an image containing text.
2. The **Pytesseract** library extracts the text from the image.
3. The **Llama 3.1** model generates a summary of the extracted text.

## 4. Visualizations and Insights
The project includes a feature that allows you to visualize:

- **Text Extraction Results:** The raw text extracted from the images.
- **Summarization Results:** A concise, high-quality summary of the extracted text.

## 5. Conclusion
This AI Text Summarizer effectively extracts text from images and provides concise summaries using the **Llama 3.1** model. By running this project in Google Colab, users can avoid potential system overloads from large language models.


