#!/bin/bash
# Download all datasets mentioned in the paper
# Paper Table II: Data Integration Layer

set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "=========================================="
echo "EAC Data Integration - Dataset Downloader"
echo "=========================================="
echo ""

# ============================================================================
# 1. USDA FoodData Central
# ============================================================================
echo "1. Downloading USDA FoodData Central..."
mkdir -p "$DATA_DIR/usda_fooddata"
cd "$DATA_DIR/usda_fooddata"

if [ ! -f "FoodData_Central_csv_2023-10-26.zip" ]; then
    echo "   Downloading FoodData Central (400K+ foods, ~1GB)..."
    curl -L "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_csv_2023-10-26.zip" -o "FoodData_Central_csv_2023-10-26.zip"
    unzip -q "FoodData_Central_csv_2023-10-26.zip"
    echo "   ✓ Downloaded and extracted"
else
    echo "   ✓ Already downloaded"
fi

cd ../..

# ============================================================================
# 2. Open Food Facts
# ============================================================================
echo ""
echo "2. Downloading Open Food Facts..."
mkdir -p "$DATA_DIR/openfoodfacts"
cd "$DATA_DIR/openfoodfacts"

if [ ! -f "en.openfoodfacts.org.products.csv" ]; then
    echo "   Downloading Open Food Facts (2M+ products, ~5GB)..."
    echo "   This may take 10-30 minutes..."
    curl -L "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv" -o "en.openfoodfacts.org.products.csv"
    echo "   ✓ Downloaded"
else
    echo "   ✓ Already downloaded"
fi

cd ../..

# ============================================================================
# 3. CDC Social Vulnerability Index
# ============================================================================
echo ""
echo "3. Downloading CDC SVI..."
mkdir -p "$DATA_DIR/cdc_svi"
cd "$DATA_DIR/cdc_svi"

if [ ! -f "SVI2022_US.csv" ]; then
    echo "   Downloading CDC SVI 2022..."
    curl -L "https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html" -o "svi_download.html"
    # Note: CDC SVI requires manual download from website
    echo "   ⚠️  CDC SVI requires manual download:"
    echo "      1. Visit: https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html"
    echo "      2. Download 'SVI2022_US.csv'"
    echo "      3. Place in: $DATA_DIR/cdc_svi/"
else
    echo "   ✓ Already downloaded"
fi

cd ../..

# ============================================================================
# 4. Area Deprivation Index
# ============================================================================
echo ""
echo "4. Downloading ADI..."
mkdir -p "$DATA_DIR/adi"
cd "$DATA_DIR/adi"

echo "   ⚠️  ADI requires registration:"
echo "      1. Visit: https://www.neighborhoodatlas.medicine.wisc.edu/"
echo "      2. Register for free account"
echo "      3. Download '2022 ADI_9 Digit Zip Code_v4.0.txt'"
echo "      4. Convert to CSV and place in: $DATA_DIR/adi/"

cd ../..

# ============================================================================
# 5. USDA Food Access Research Atlas
# ============================================================================
echo ""
echo "5. Downloading USDA Food Access Research Atlas..."
mkdir -p "$DATA_DIR/usda_food_atlas"
cd "$DATA_DIR/usda_food_atlas"

if [ ! -f "FoodAccessResearchAtlasData2019.xlsx" ]; then
    echo "   Downloading Food Access Atlas..."
    curl -L "https://www.ers.usda.gov/webdocs/DataFiles/80591/FoodAccessResearchAtlasData2019.xlsx" -o "FoodAccessResearchAtlasData2019.xlsx"
    echo "   ✓ Downloaded (convert to CSV manually)"
else
    echo "   ✓ Already downloaded"
fi

cd ../..

# ============================================================================
# 6. Instacart Market Basket
# ============================================================================
echo ""
echo "6. Downloading Instacart dataset..."
mkdir -p "$DATA_DIR/instacart"
cd "$DATA_DIR/instacart"

echo "   ⚠️  Instacart requires Kaggle account:"
echo "      1. Install kaggle CLI: pip install kaggle"
echo "      2. Setup API key: https://www.kaggle.com/docs/api"
echo "      3. Run: kaggle competitions download -c instacart-market-basket-analysis"
echo "      4. Unzip in: $DATA_DIR/instacart/"

cd ../..

# ============================================================================
# 7. dunnhumby Complete Journey
# ============================================================================
echo ""
echo "7. Downloading dunnhumby dataset..."
mkdir -p "$DATA_DIR/dunnhumby"
cd "$DATA_DIR/dunnhumby"

echo "   ⚠️  dunnhumby requires registration:"
echo "      1. Visit: https://www.dunnhumby.com/source-files/"
echo "      2. Register and accept terms"
echo "      3. Download 'The Complete Journey'"
echo "      4. Place in: $DATA_DIR/dunnhumby/"

cd ../..

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "Download Summary"
echo "=========================================="
echo "✓ Automated downloads complete"
echo "⚠️  Manual downloads required for:"
echo "   - CDC SVI (registration required)"
echo "   - ADI (registration required)"
echo "   - Instacart (Kaggle account required)"
echo "   - dunnhumby (registration required)"
echo ""
echo "Next steps:"
echo "1. Complete manual downloads"
echo "2. Run: python scripts/prepare_datasets.py"
echo "3. Verify: python scripts/verify_data.py"
echo ""