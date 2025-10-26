# EAC Checkout Simulator - React Frontend

Beautiful, interactive checkout simulator for the EAC Agent system.

## Features

- 🛒 **Interactive Cart** - Add/remove items, see real-time totals
- ✨ **Smart Recommendations** - AI-powered product suggestions
- 📊 **Impact Dashboard** - Visualize savings and nutrition improvements
- 👤 **User Profiles** - Customize SDOH and demographic factors
- 🎨 **Modern UI** - Built with React + TailwindCSS

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

The app will be available at http://localhost:3000

## Prerequisites

Make sure the EAC API is running:

```bash
# In the project root
uvicorn api.main:app --reload
```

## Architecture

```
src/
├── App.jsx              # Main app component
├── components/
│   ├── Cart.jsx         # Shopping cart view
│   ├── Recommendations.jsx  # Recommendation cards
│   ├── Impact.jsx       # Results/impact dashboard
│   └── UserProfile.jsx  # User profile editor
├── main.jsx             # Entry point
└── index.css            # Global styles
```

## Usage

1. **Adjust User Profile** - Set income, household size, SNAP eligibility, etc.
2. **Review Cart** - See pre-loaded items or add your own
3. **Checkout** - Click "Proceed to Checkout" to get recommendations
4. **Accept/Decline** - Review each recommendation and make choices
5. **See Impact** - View your savings, nutrition improvements, and insights

## API Integration

The app connects to the FastAPI backend at `http://localhost:8000/api/checkout`.

Mock data is used as fallback if the API is unavailable.

## Customization

### Add More Products

Edit the initial cart in `src/App.jsx`:

```javascript
const [cart, setCart] = useState([
  { id: 1, name: 'Your Product', price: 4.99, category: 'Category', hei: 45 },
  // Add more...
])
```

### Change Theme Colors

Edit `tailwind.config.js`:

```javascript
theme: {
  extend: {
    colors: {
      primary: '#10b981',  // Green
      secondary: '#3b82f6', // Blue
    }
  }
}
```

## Deployment

### Netlify

```bash
npm run build
# Deploy the dist/ folder
```

### Vercel

```bash
vercel deploy
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "preview"]
```

## Screenshots

### Cart View
![Cart](docs/cart.png)

### Recommendations
![Recommendations](docs/recommendations.png)

### Impact Dashboard
![Impact](docs/impact.png)

## License

MIT
