import { useState } from 'react'
import Cart from './components/Cart'
import Recommendations from './components/Recommendations'
import Impact from './components/Impact'
import UserProfile from './components/UserProfile'
import { ShoppingCart, Sparkles, TrendingUp } from 'lucide-react'

function App() {
  const [step, setStep] = useState('cart') // cart, recommendations, impact
  const [cart, setCart] = useState([
    { id: 1, name: 'Whole Milk (1 gal)', price: 4.99, category: 'Dairy', hei: 45 },
    { id: 2, name: 'White Bread', price: 2.49, category: 'Bakery', hei: 35 },
    { id: 3, name: 'Soda (12pk)', price: 5.99, category: 'Beverages', hei: 20 },
    { id: 4, name: 'Potato Chips', price: 3.49, category: 'Snacks', hei: 25 },
    { id: 5, name: 'Ground Beef (1lb)', price: 6.99, category: 'Meat', hei: 50 },
  ])
  const [recommendations, setRecommendations] = useState([])
  const [acceptedRecs, setAcceptedRecs] = useState([])
  const [userProfile, setUserProfile] = useState({
    income: 35000,
    snap_eligible: true,
    food_insecurity: 0.7,
    diabetes_risk: 0.4,
    household_size: 3
  })

  const handleCheckout = async () => {
    // Call EAC API
    try {
      const response = await fetch('/api/v1/checkout/decide', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'demo_user',
          cart: cart.map(item => ({
            product_id: item.id.toString(),
            quantity: 1,
            price: item.price
          })),
          delivery_address: {
            zip_code: userProfile.snap_eligible ? '94102' : '94301',
            census_tract: '06075017902'
          },
          payment_methods: userProfile.snap_eligible ? ['SNAP_EBT', 'CREDIT_CARD'] : ['CREDIT_CARD'],
          consent: {
            personalization: true,
            sdoh_signals: true
          },
          metadata: {
            income: userProfile.income,
            food_insecurity: userProfile.food_insecurity,
            diabetes_risk: userProfile.diabetes_risk,
            household_size: userProfile.household_size
          }
        })
      })
      
      const data = await response.json()
      
      // Map API response to frontend format
      const mappedRecs = data.recommendations?.map((rec, idx) => ({
        id: `r${idx + 1}`,
        original_product: rec.original_product_name,
        suggested_product: rec.suggested_product_name,
        original_price: cart.find(item => item.name === rec.original_product_name)?.price || 0,
        suggested_price: cart.find(item => item.name === rec.original_product_name)?.price - rec.savings || 0,
        savings: rec.savings,
        nutrition_improvement: rec.nutrition_improvement,
        reason: rec.reason,
        confidence: rec.confidence
      })) || []
      
      setRecommendations(mappedRecs.length > 0 ? mappedRecs : [
        {
          id: 'r1',
          original_product: 'Whole Milk (1 gal)',
          suggested_product: '2% Milk (1 gal)',
          original_price: 4.99,
          suggested_price: 3.49,
          savings: 1.50,
          nutrition_improvement: 10,
          reason: 'Lower fat, SNAP eligible',
          confidence: 0.85
        },
        {
          id: 'r2',
          original_product: 'White Bread',
          suggested_product: 'Whole Wheat Bread',
          original_price: 2.49,
          suggested_price: 1.99,
          savings: 0.50,
          nutrition_improvement: 15,
          reason: 'More fiber, better for blood sugar',
          confidence: 0.78
        },
        {
          id: 'r3',
          original_product: 'Soda (12pk)',
          suggested_product: 'Sparkling Water (12pk)',
          original_price: 5.99,
          suggested_price: 3.99,
          savings: 2.00,
          nutrition_improvement: 25,
          reason: 'Zero sugar, SNAP eligible',
          confidence: 0.92
        }
      ])
      setStep('recommendations')
    } catch (error) {
      console.error('Checkout error:', error)
      // Use mock data on error
      setRecommendations([
        {
          id: 'r1',
          original_product: 'Whole Milk (1 gal)',
          suggested_product: '2% Milk (1 gal)',
          original_price: 4.99,
          suggested_price: 3.49,
          savings: 1.50,
          nutrition_improvement: 10,
          reason: 'Lower fat, SNAP eligible',
          confidence: 0.85
        },
        {
          id: 'r2',
          original_product: 'White Bread',
          suggested_product: 'Whole Wheat Bread',
          original_price: 2.49,
          suggested_price: 1.99,
          savings: 0.50,
          nutrition_improvement: 15,
          reason: 'More fiber, better for blood sugar',
          confidence: 0.78
        },
        {
          id: 'r3',
          original_product: 'Soda (12pk)',
          suggested_product: 'Sparkling Water (12pk)',
          original_price: 5.99,
          suggested_price: 3.99,
          savings: 2.00,
          nutrition_improvement: 25,
          reason: 'Zero sugar, SNAP eligible',
          confidence: 0.92
        }
      ])
      setStep('recommendations')
    }
  }

  const handleAccept = async (recId) => {
    const rec = recommendations.find(r => r.id === recId)
    setAcceptedRecs([...acceptedRecs, rec])
    
    // Send feedback to backend
    try {
      await fetch('/api/v1/checkout/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'demo_user',
          transaction_id: `txn_${Date.now()}`,
          total_recommendations: recommendations.length,
          accepted_count: acceptedRecs.length + 1,
          total_savings: rec.savings,
          nutrition_improvement: rec.nutrition_improvement,
          fairness_violation: false
        })
      })
      console.log('✓ Feedback sent: Accepted', rec.suggested_product)
    } catch (error) {
      console.log('⚠️ Feedback failed (backend may be offline)')
    }
  }

  const handleDecline = async (recId) => {
    const rec = recommendations.find(r => r.id === recId)
    setRecommendations(recommendations.filter(r => r.id !== recId))
    
    // Send feedback to backend
    try {
      await fetch('/api/v1/checkout/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'demo_user',
          transaction_id: `txn_${Date.now()}`,
          total_recommendations: recommendations.length,
          accepted_count: acceptedRecs.length,
          total_savings: 0,
          nutrition_improvement: 0,
          fairness_violation: false
        })
      })
      console.log('✓ Feedback sent: Declined', rec.suggested_product)
    } catch (error) {
      console.log('⚠️ Feedback failed (backend may be offline)')
    }
  }

  const handleComplete = () => {
    setStep('impact')
  }

  const handleReset = () => {
    setStep('cart')
    setRecommendations([])
    setAcceptedRecs([])
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-green-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Sparkles className="h-8 w-8 text-primary" />
              <h1 className="text-2xl font-bold text-gray-900">
                EAC Checkout Simulator
              </h1>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                step === 'cart' ? 'bg-primary text-white' : 'bg-gray-200 text-gray-600'
              }`}>
                <ShoppingCart className="inline h-4 w-4 mr-1" />
                Cart
              </div>
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                step === 'recommendations' ? 'bg-primary text-white' : 'bg-gray-200 text-gray-600'
              }`}>
                <Sparkles className="inline h-4 w-4 mr-1" />
                Recommendations
              </div>
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                step === 'impact' ? 'bg-primary text-white' : 'bg-gray-200 text-gray-600'
              }`}>
                <TrendingUp className="inline h-4 w-4 mr-1" />
                Impact
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - User Profile */}
          <div className="lg:col-span-1">
            <UserProfile profile={userProfile} setProfile={setUserProfile} />
          </div>

          {/* Right Column - Main Flow */}
          <div className="lg:col-span-2">
            {step === 'cart' && (
              <Cart 
                cart={cart} 
                setCart={setCart} 
                onCheckout={handleCheckout} 
              />
            )}
            
            {step === 'recommendations' && (
              <Recommendations
                recommendations={recommendations}
                onAccept={handleAccept}
                onDecline={handleDecline}
                onComplete={handleComplete}
              />
            )}
            
            {step === 'impact' && (
              <Impact
                acceptedRecs={acceptedRecs}
                onReset={handleReset}
              />
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
