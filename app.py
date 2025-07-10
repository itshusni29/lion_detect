import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import pandas as pd  # Import pandas here

# Load the YOLOv8 model
# If the model is large, consider downloading it dynamically (e.g., from Google Drive or AWS S3)
model = YOLO("best.pt")  # Load your model

# Title for the Streamlit App
st.title("YOLOv8 Object Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show spinner while processing
    with st.spinner('Processing image...'):
        # Open the image using PIL
        image = Image.open(uploaded_file)

        # Convert image to a format acceptable by the model (OpenCV format)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV

        # Make predictions using YOLOv8 model
        results = model(image_cv)

        # Extract the first result (the prediction result)
        result = results[0]  # Get the first result object

        # Plot the result (this draws bounding boxes on the image)
        output_image = result.plot()  # This image is in RGB format

        # Convert back the resulting image from BGR (if necessary) to RGB before displaying it
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)  # Convert back to RGB format

        # Display the image with bounding boxes in Streamlit
        st.image(output_image_rgb, caption="Predicted Image", use_column_width=True)

        # Convert the results to a DataFrame
        df = result.to_df()

        # Extract bounding box coordinates from the 'box' column
        df[['xmin', 'ymin', 'xmax', 'ymax']] = pd.DataFrame(df['box'].to_list(), index=df.index)

        # Show the DataFrame with the predicted bounding boxes and other details (like class and confidence)
        st.write("Prediction Results (Bounding boxes, Confidence, Class):")
        st.write(df[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name']])  # Show relevant columns

    st.success('Prediction complete!')  # Show success message after processing
