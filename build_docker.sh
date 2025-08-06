#!/bin/bash

# Build script for TinyLlama Docker images

set -e

echo "🐳 Building TinyLlama Docker Images"
echo "==================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo ""
echo "📋 Available build options:"
echo "1. CPU-only image (smaller, works everywhere)"
echo "2. GPU-enabled image (CUDA support)"
echo "3. Both images"
echo ""

read -p "Choose option (1-3): " choice

case $choice in
    1)
        echo "🔨 Building CPU-only image..."
        docker build -f Dockerfile.cpu -t tinyllama-api:cpu .
        echo "✅ CPU image built successfully!"
        echo "🚀 Run with: docker run -p 8000:8000 tinyllama-api:cpu"
        ;;
    2)
        echo "🔨 Building GPU-enabled image..."
        docker build -f Dockerfile -t tinyllama-api:gpu .
        echo "✅ GPU image built successfully!"
        echo "🚀 Run with: docker run --gpus all -p 8000:8000 tinyllama-api:gpu"
        ;;
    3)
        echo "🔨 Building both images..."
        echo "Building CPU image..."
        docker build -f Dockerfile.cpu -t tinyllama-api:cpu .
        echo "Building GPU image..."
        docker build -f Dockerfile -t tinyllama-api:gpu .
        echo "✅ Both images built successfully!"
        echo "🚀 CPU: docker run -p 8000:8000 tinyllama-api:cpu"
        echo "🚀 GPU: docker run --gpus all -p 8000:8000 tinyllama-api:gpu"
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "📦 Available images:"
docker images | grep tinyllama-api

echo ""
echo "💡 Tips:"
echo "- The model is embedded in the image (~2.5GB total)"
echo "- No internet connection needed for inference"
echo "- Use docker-compose for easier deployment"
echo "- Test with: curl http://localhost:8000/health"