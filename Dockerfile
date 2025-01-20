# Use the latest official PyTorch runtime
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

# Set environment variables

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive  
ENV HOME /app
# Configure matplotlib to use temporary directory
ENV MPLCONFIGDIR /tmp/matplotlib 

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    awscli \
    wget \
    libgl1-mesa-glx \ 
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# Install MMEngine, MMCV and MMDet
RUN pip install --no-cache-dir openmim && \
    mim install --no-cache-dir "mmengine>=0.6.0" "mmcv>=2.0.0rc4,<2.1.0" "mmdet>=3.0.0,<4.0.0"
RUN pip install --no-cache-dir opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64

# Install MMYOLO
RUN git clone https://github.com/open-mmlab/mmyolo.git -b dev mmyolo \
    && cd mmyolo \
    && mim install --no-cache-dir -e . \
    && cd ..

# Download model weights
# NOTE: Replace this section with instructions for users to download their own models
# or provide a separate script to download models from a configurable source
ENV MODELS_URL="your-models-url"
ENV WEIGHTS_URL="https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco/rtmdet_l_syncbn_fast_8xb32-300e_coco_20230102_135928-ee3abdc4.pth"

RUN mkdir -p /app/weights && \
    wget -P /app/weights ${WEIGHTS_URL}

# Install TinySAM
RUN git clone https://github.com/xinghaochen/TinySAM.git /app/TinySAM

# NOTE: Users need to provide their own TinySAM weights
ENV TINYSAM_WEIGHTS_PATH="/app/TinySAM/weights/tinysam.pth"

# Set permissions
RUN chmod -R 755 /app/

# Copy application code
COPY . /app/

# Set up serve script
COPY serve /usr/bin/serve
RUN chmod +x /usr/bin/serve

# Expose port for the Flask application
EXPOSE 8080

# Start the application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
