# How to Capture Screenshots for README

This guide will help you capture the screenshots referenced in the README.

---

## 📁 **Screenshot Directory**

Create the directory first:
```bash
mkdir -p docs/screenshots
```

---

## 📸 **Screenshots Needed**

### **Frontend Screenshots (3 total)**

1. **`frontend-cart.png`** - Shopping Cart View
2. **`frontend-recommendations.png`** - Recommendations View  
3. **`frontend-impact.png`** - Impact Dashboard View

### **Dashboard Screenshots (4 total)**

4. **`dashboard-overview.png`** - Dashboard Overview
5. **`dashboard-charts.png`** - Performance Charts
6. **`dashboard-fairness.png`** - Fairness Analysis
7. **`dashboard-transactions.png`** - Transaction Table

---

## 🚀 **Step-by-Step Capture Guide**

### **Preparation**

1. **Start all services:**
```bash
# Terminal 1: API
source .venv/bin/activate
uvicorn api.main:app --reload

# Terminal 2: Frontend
cd frontend/react-app
npm run dev

# Terminal 3: Dashboard
streamlit run frontend/streamlit_dashboard.py
```

2. **Create some test data:**
   - Open http://localhost:3000
   - Do 3-5 checkout transactions
   - Accept some recommendations, decline others
   - This will populate the analytics dashboard

---

## 📷 **Frontend Screenshots**

### **1. frontend-cart.png**

**What to capture:**
- Open http://localhost:3000
- Adjust user profile to show interesting values:
  - Income: $35,000
  - SNAP Eligible: ✓
  - Food Insecurity: 70%
  - Diabetes Risk: 40%
- Cart should have 5 items visible
- Make sure "Proceed to Checkout" button is visible

**How to capture (macOS):**
```bash
Cmd + Shift + 4
# Drag to select the browser window
# File will be saved to Desktop
# Rename to frontend-cart.png
# Move to docs/screenshots/
```

---

### **2. frontend-recommendations.png**

**What to capture:**
- Click "Proceed to Checkout"
- Wait for recommendations to load
- Capture the recommendations view showing:
  - 2-3 recommendation cards
  - Savings amounts visible
  - Nutrition improvements visible
  - Accept/Decline buttons visible
  - Confidence scores visible

**Tip:** Don't click Accept/Decline yet - capture first!

---

### **3. frontend-impact.png**

**What to capture:**
- Accept 2 out of 3 recommendations
- Click "Complete Purchase"
- Capture the impact view showing:
  - Total savings (green box)
  - Nutrition improvement (red box)
  - Processing time (blue box)
  - List of accepted recommendations
  - Insights section

---

## 📊 **Dashboard Screenshots**

### **4. dashboard-overview.png**

**What to capture:**
- Open http://localhost:8501
- Click "🔄 Refresh Data" to load latest transactions
- Scroll to top
- Capture showing:
  - Title "EAC Analytics Dashboard"
  - 4 key metrics boxes (Acceptance Rate, Avg Savings, Nutrition, Latency)
  - First chart (Acceptance Rate by Policy)

---

### **5. dashboard-charts.png**

**What to capture:**
- Scroll down to show the charts section
- Capture showing:
  - "Performance Analysis" header
  - Acceptance Rate by Policy (bar chart)
  - Savings Distribution (histogram)
  - Nutrition Impact by Policy (box plot)
  - Latency Over Time (line chart)

**Tip:** You may need to capture this in two parts and stitch together, or zoom out browser to 75%

---

### **6. dashboard-fairness.png**

**What to capture:**
- Scroll to "Fairness Analysis" section
- Capture showing:
  - "Fairness Analysis" header
  - Savings by Group (bar chart)
  - Acceptance Rate by Group (bar chart)
  - Fairness Check status (PASS/REVIEW)

---

### **7. dashboard-transactions.png**

**What to capture:**
- Scroll to "Recent Transactions" section
- Capture showing:
  - "Recent Transactions" header
  - Transaction table with at least 5-10 rows
  - Column headers visible (user_id, policy_used, delta_spend, etc.)
  - "Download Full Dataset" button visible

---

## 🎨 **Screenshot Best Practices**

### **Resolution**
- Use at least 1920x1080 display
- Capture at 100% browser zoom
- Full window capture (not just content)

### **Browser**
- Use Chrome or Firefox
- Hide bookmarks bar for cleaner look
- Close unnecessary tabs

### **Content**
- Use realistic data (not empty states)
- Show positive numbers (savings, improvements)
- Ensure all text is readable
- No personal information visible

### **File Format**
- Save as PNG (not JPG)
- Don't compress too much
- Keep file size under 500KB if possible

---

## 🖼️ **After Capturing**

### **1. Move files to correct location:**
```bash
mv ~/Desktop/frontend-cart.png docs/screenshots/
mv ~/Desktop/frontend-recommendations.png docs/screenshots/
mv ~/Desktop/frontend-impact.png docs/screenshots/
mv ~/Desktop/dashboard-overview.png docs/screenshots/
mv ~/Desktop/dashboard-charts.png docs/screenshots/
mv ~/Desktop/dashboard-fairness.png docs/screenshots/
mv ~/Desktop/dashboard-transactions.png docs/screenshots/
```

### **2. Verify all screenshots:**
```bash
ls -lh docs/screenshots/
```

You should see 7 PNG files.

### **3. Commit to git:**
```bash
git add docs/screenshots/
git commit -m "Add screenshots for README documentation"
git push origin main
```

---

## ✅ **Checklist**

Before committing, verify:

- [ ] All 7 screenshots captured
- [ ] Files are in `docs/screenshots/` directory
- [ ] Files are named correctly (exact names from list)
- [ ] All screenshots are PNG format
- [ ] Images are clear and readable
- [ ] No personal/sensitive information visible
- [ ] File sizes are reasonable (< 500KB each)

---

## 🎯 **Quick Capture Commands**

**macOS:**
```bash
# Selected area
Cmd + Shift + 4

# Specific window
Cmd + Shift + 4, then Space, click window

# Full screen
Cmd + Shift + 3
```

**Windows:**
```bash
# Snipping tool
Windows + Shift + S

# Full screen
PrtScn

# Active window
Alt + PrtScn
```

**Linux:**
```bash
# GNOME
PrtScn or Shift + PrtScn

# KDE
Spectacle
```

---

## 📝 **Notes**

- Screenshots will appear in README.md automatically once added to `docs/screenshots/`
- GitHub will render the images when viewing the README
- If images don't show, check file paths and names match exactly
- You can update screenshots anytime by replacing the files

---

**Ready to capture? Start your services and follow the guide above!** 📸
