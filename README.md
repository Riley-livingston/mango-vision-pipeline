# 🥭 Visual Search Service

A containerized machine learning service for visual search capabilities using PyTorch, MMYolo, and TinySAM.

## Overview

This service provides visual search functionality using state-of-the-art deep learning models. It's containerized using Docker and designed to be deployed on AWS, though it can be adapted for other cloud providers or on-premises deployment.

## Prerequisites

* Docker
* AWS Account (if deploying to AWS)
* Python 3.8+
* CUDA-capable GPU (recommended for production)

## Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```plaintext
AWS_ACCOUNT_ID=your_aws_account_id
AWS_REGION=your_aws_region
ECR_REPO_NAME=your_ecr_repo_name
```

### Model Weights

Before building the container, you'll need to:

1. Download the MMYOLO weights from the official repository
2. Obtain TinySAM weights and place them in the appropriate directory
3. Configure any custom model weights

## Local Development

1. Clone the repository:
    ```bash
    git clone https://github.com/riley-livingston/mango-vision-pipeline.git
    cd mango-vision-pipeline
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up model weights:
    ```bash
    # Download MMYOLO weights
    wget -P weights/ https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco/rtmdet_l_syncbn_fast_8xb32-300e_coco_20230102_135928-ee3abdc4.pth

    # Place your TinySAM weights in the appropriate directory
    cp path/to/your/tinysam.pth TinySAM/weights/
    ```

## Docker Build and Run

Build the Docker image:
```bash
docker build -t visual-search-service .
```

## AWS Deployment

### Prerequisites

1. Install and configure the AWS CLI
2. Create an ECR repository
3. Set up AWS CodeBuild project

### Deployment Steps

1. Configure AWS credentials:
    ```bash
    aws configure
    ```

2. Update the buildspec.yml with your AWS details

3. Push to your repository to trigger the CI/CD pipeline

### Manual Deployment

You can also deploy manually using:

```bash
# Login to ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build and tag
docker build -t ${ECR_REPO_NAME} .
docker tag ${ECR_REPO_NAME}:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:latest

# Push
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:latest
```

## API Documentation

The service exposes a REST API on port 8080:

* `POST /predict`: Submit an image for processing
  * Accepts: multipart/form-data with an image file
  * Returns: JSON with prediction results

Example usage:
```bash
curl -X POST -F "image=@/path/to/image.jpg" http://localhost:8080/predict
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* [MMYOLO](https://github.com/open-mmlab/mmyolo)
* [TinySAM](https://github.com/xinghaochen/TinySAM)
* PyTorch team
