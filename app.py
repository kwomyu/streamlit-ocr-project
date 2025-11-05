import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pytesseract
import cv2
import numpy as np

# --- THIS IS FOR WINDOWS USERS ---
# You MUST tell pytesseract where you installed Tesseract
# Uncomment this line and change the path if you're on Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(img, is_canvas=False):
    """
    Converts a PIL Image to grayscale, applies preprocessing,
    and ensures the result is WHITE TEXT on a BLACK BACKGROUND.
    """
    # Convert PIL Image to an OpenCV image (and ensure it's RGB)
    img_cv = np.array(img.convert('RGB'))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    if is_canvas:
        # For canvas: the drawing is already white on black
        # Apply a simple threshold to clean it up
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours to crop to the drawn content
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of all contours combined
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = 0, 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            
            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(thresh.shape[1], x_max + padding)
            y_max = min(thresh.shape[0], y_max + padding)
            
            # Crop to the drawn area
            thresh = thresh[y_min:y_max, x_min:x_max]
            
            # Resize to a standard size (helps with recognition)
            # Make it square
            size = max(thresh.shape[0], thresh.shape[1])
            square = np.zeros((size, size), dtype=np.uint8)
            y_offset = (size - thresh.shape[0]) // 2
            x_offset = (size - thresh.shape[1]) // 2
            square[y_offset:y_offset+thresh.shape[0], x_offset:x_offset+thresh.shape[1]] = thresh
            
            # Resize to a reasonable size
            thresh = cv2.resize(square, (200, 200), interpolation=cv2.INTER_AREA)
        
        return thresh
    else:
        # For uploaded images and webcam: use adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Invert for white text on black background
        inverted_thresh = cv2.bitwise_not(thresh)
        
        return inverted_thresh

def get_tesseract_data(img, psm_mode):
    """Runs Tesseract on a preprocessed image and returns the data dictionary."""
    config = f'--psm {psm_mode}'
    data = pytesseract.image_to_data(img, lang='eng', config=config, output_type=pytesseract.Output.DICT)
    return data

def format_ocr_output(data, confidence_threshold):
    """Formats the Tesseract data into an HTML string with confidence highlighting."""
    formatted_text = ""
    total_confidence = 0
    word_count = 0

    for i in range(len(data['text'])):
        word = data['text'][i]
        conf = int(data['conf'][i])

        if conf != -1 and word.strip():
            total_confidence += conf
            word_count += 1

            if conf < confidence_threshold:
                formatted_text += f'<span style="background-color: #FFFF00; padding: 2px;">{word}</span> '
            else:
                formatted_text += f'{word} '

    if word_count > 0:
        avg_confidence = total_confidence / word_count
        return formatted_text, avg_confidence
    else:
        return "No text found.", 0

# Set page title
st.title("Kwomyu's OCR Project 📄")
st.write("Use Google's Tesseract engine to extract text from any image.")

# --- 1. SET UP THE INPUTS ---
st.header("Upload, Draw, or Snap a Pic")
input_mode = st.radio("Choose your input method:", ["Upload an Image", "Draw a Character", "Use Webcam"])

image = None
is_canvas = False

if input_mode == "Upload an Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif input_mode == "Draw a Character":
    st.write("Draw a single letter or digit (white on black works best!)")
    st.info("💡 Tip: Draw large and clear. The character will be automatically cropped and centered.")
    
    # Create a canvas
    canvas_result = st_canvas(
        fill_color="rgb(0, 0, 0)",
        stroke_width=25,
        stroke_color="rgb(255, 255, 255)",
        background_color="rgb(0, 0, 0)",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if canvas_result.image_data is not None:
        # Check if something was actually drawn
        img_data = canvas_result.image_data.astype(np.uint8)
        # Check if there's any white pixels
        if np.any(img_data[:, :, :3] > 10):
            image = Image.fromarray(img_data[:, :, :3])
            is_canvas = True

elif input_mode == "Use Webcam":
    st.info("💡 Tip: Hold the text steady and ensure good lighting for best results.")
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)

# --- 2. SHOW THE INPUT IMAGE ---
if image is not None:
    st.subheader("Your Image:")
    st.image(image, width='stretch')
else:
    st.info("Please provide an image to get started.")

# --- 3. RUN OCR AND SHOW RESULTS ---
st.header("Results")

# Add a confidence threshold slider
conf_threshold = st.slider("Confidence Highlight Threshold", 0, 100, 60, 
                          help="Words with confidence below this will be highlighted.")

# PSM Mode Selector
psm_options = {
    "Assume a full page (Default)": 3,
    "Assume a single block of text": 6,
    "Treat as a single line of text": 7,
    "Treat as a single character": 10
}

psm_selection = st.selectbox(
    "Tesseract Page Mode (Advanced)", 
    options=list(psm_options.keys()),
    index=3 if input_mode == "Draw a Character" else 0,  # Default to single character for drawing
    help="This tells Tesseract what kind of image it's looking at."
)

if st.button("Extract Text") and image is not None:
    with st.spinner("Extracting text... this might take a moment."):
        
        # 1. Determine the PSM mode
        if input_mode == "Draw a Character":
            psm = 10  # Force PSM 10 for drawing
            st.info("Using 'Single Character' mode (PSM 10) for drawn image.")
        else:
            psm = psm_options[psm_selection]

        # 2. Preprocess the image
        preprocessed_img = preprocess_image(image, is_canvas=is_canvas)
        
        # Show the preprocessed image
        with st.expander("🔍 View Preprocessed Image (sent to Tesseract)"):
            st.image(preprocessed_img, caption="Preprocessed Image", width='stretch')
            
        # 3. Get Tesseract data
        try:
            ocr_data = get_tesseract_data(preprocessed_img, psm_mode=psm)

            # 4. Format and display the output
            formatted_text, avg_confidence = format_ocr_output(ocr_data, conf_threshold)

            st.subheader("Average Confidence: {:.2f}%".format(avg_confidence))

            # Check for auto-confirm
            if avg_confidence > 90:
                st.success("✅ Auto-confirmed: High confidence result.")
            elif avg_confidence > 0:
                st.warning("⚠️ Medium confidence - review highlighted words.")
            else:
                st.error("❌ No text detected. Try adjusting the image or PSM mode.")

            st.subheader("Highlighted Text:")
            st.markdown(formatted_text, unsafe_allow_html=True)

            # Show the raw data in an expander
            with st.expander("Show Raw Tesseract Data"):
                st.json(ocr_data)
                
        except Exception as e:
            st.error(f"Error during OCR processing: {str(e)}")
            st.info("Try adjusting the PSM mode or improving image quality.")