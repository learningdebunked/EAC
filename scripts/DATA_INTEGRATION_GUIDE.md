# Data Integration Guide

Complete guide for integrating all real-world datasets mentioned in the paper.

## Overview

The EAC system requires 7 primary datasets across 3 categories:

### **A) SDOH & Equity Data** (5 datasets)
1. CDC Social Vulnerability Index (SVI)
2. Area Deprivation Index (ADI)
3. USDA Food Access Research Atlas
4. National Transit Map
5. U.S. Census ACS

### **B) Product & Nutrition Data** (2 datasets)
6. USDA FoodData Central
7. Open Food Facts

### **C) Transaction Data** (2 datasets)
8. Instacart Market Basket
9. dunnhumby Complete Journey

---

## Quick Start

```bash
# 1. Run automated downloader
chmod +x scripts/download_datasets.sh
./scripts/download_datasets.sh

# 2. Complete manual downloads (see below)

# 3. Prepare datasets
python scripts/prepare_datasets.py

# 4. Verify integration
python scripts/verify_data.py

# 5. Run with real data
python examples/run_simulation.py --use-real-data