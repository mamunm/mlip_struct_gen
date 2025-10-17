#!/bin/bash
# Build script for C++ analysis extension

set -e  # Exit on error

echo "Building C++ analysis extension..."

# Navigate to C++ directory
cd mlip_struct_gen/analysis/cpp

# Create build directory
mkdir -p build
cd build

# Configure and build
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "Building..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    make -j$(sysctl -n hw.ncpu)
else
    # Linux
    make -j$(nproc)
fi

# Copy to analysis directory
echo "Installing extension..."
cp _analysis_core*.so ../../

echo "Build complete!"
echo "Extension installed at: mlip_struct_gen/analysis/_analysis_core*.so"

# Go back to project root
cd ../../../..

# Test import
echo "Testing import..."
python -c "from mlip_struct_gen.analysis import RDF; print('✓ Extension successfully installed and working!')" || echo "✗ Import failed - extension may not be properly compiled"
