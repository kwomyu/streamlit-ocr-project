Kwomyu's Multi-Mode OCR Project

This is a robust web application built with Python and Streamlit that uses Google's industry-standard Tesseract OCR engine to extract text from images. This project is engineered for flexibility, allowing the user to precisely control the recognition process for complex input types.Built to showcase my skills in machine learning utility development, image preprocessing, and reliable system design.


I see the problem! When you copy from the chat, even the plain Markdown can sometimes lose its spacing and formatting, especially inside tables.

Here is the "smarter way"—the plaintext method. This uses simple formatting that is guaranteed to paste correctly into any editor or file. I'll use bolding and separation instead of a complex table structure.

Just copy the text below and replace the old table section in your README.md.

✨ Core Features & Advanced Engineering
1. Custom Canvas Recognition Pipeline (The Smart Fix for Hand-Drawing!)
Engineering Solution: For hand-drawn input, the app automatically uses OpenCV to detect the character's contours, then crops it tightly, centers it, and resizes it to a clean 200x200 image.

Portfolio Highlight: This demonstrates Image Manipulation (OpenCV) and problem-solving to overcome Tesseract's known limitations with messy, uncropped input, dramatically increasing hand-drawn character accuracy.

2. Confidence-Aware User Interface (Reliability Check)
Engineering Solution: The app calculates the average confidence of the recognized text. Any words falling below the user-set threshold (default 60%) are visually highlighted in yellow.

Portfolio Highlight: This proves focus on System Reliability—the app is designed to explicitly warn the user when the AI's guess is weak (e.g., catching when 'Z' is misread as 'VA').

3. Advanced Tesseract Mode Selection
Engineering Solution: Provides the user a selector for the four most useful Tesseract Page Segmentation Modes (PSM 3, 6, 7, 10), enabling precision OCR.

Portfolio Highlight: This shows understanding of AI Tool Configuration and allows the user to correct complex problems (e.g., using PSM 10 to read a single, large letter that other modes would ignore).

4. Multi-Mode Input & Versatility
Engineering Solution: Accepts file uploads, live webcam photos, and direct drawing via the Streamlit Canvas. Uploaded photos use a smarter Adaptive Threshold to handle shadows and uneven lighting.

Portfolio Highlight: Demonstrates Versatility and practical application across different data sources, ensuring the best preprocessing is applied to each input type.

Setup and Installation

1. Install Tesseract Engine (Required)Tesseract must be installed directly on your operating system.
Windows: Download the installer from the UB-Mannheim Tesseract Wiki.
Mac: brew install tesseract
Linux (Debian/Ubuntu): sudo apt-get install tesseract-ocr

2. Set Up the Python Environment

Bash
# Clone the repository and navigate into the folder
git clone [Your-GitHub-Repo-URL-Here]
cd My-OCR-App

# Create and activate the virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate # Mac/Linux

# Install the required Python libraries
pip install streamlit streamlit-drawable-canvas pytesseract pillow opencv-python

3. Final Windows Configuration
For Windows Users ONLY: You must uncomment and verify the path to your Tesseract executable in the app.py file:

Python
# Uncomment this line and change the path if you're on Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

4. Run the ApplicationWith your virtual environment active, run the Streamlit server:Bashstreamlit run app.py