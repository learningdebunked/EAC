import { TrendingUp, DollarSign, Heart, Clock, RotateCcw, CheckCircle } from 'lucide-react'

export default function Impact({ acceptedRecs, onReset }) {
  const totalSavings = acceptedRecs.reduce((sum, rec) => sum + rec.savings, 0)
  const totalNutrition = acceptedRecs.reduce((sum, rec) => sum + rec.nutrition_improvement, 0)
  const avgLatency = 2.3 // Mock latency

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900 flex items-center">
          <TrendingUp className="mr-2 h-6 w-6 text-primary" />
          Your Impact
        </h2>
        <CheckCircle className="h-8 w-8 text-primary" />
      </div>

      {/* Success Message */}
      <div className="bg-green-50 border-2 border-primary rounded-lg p-6 mb-6">
        <h3 className="text-xl font-bold text-primary mb-2">
          🎉 Purchase Complete!
        </h3>
        <p className="text-gray-700">
          You accepted {acceptedRecs.length} recommendation{acceptedRecs.length !== 1 ? 's' : ''} and made a positive impact on your health and budget.
        </p>
      </div>

      {/* Impact Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-lg border-2 border-green-200">
          <div className="flex items-center justify-between mb-2">
            <DollarSign className="h-8 w-8 text-green-600" />
            <span className="text-sm font-medium text-green-700">Savings</span>
          </div>
          <p className="text-3xl font-bold text-green-600">
            ${totalSavings.toFixed(2)}
          </p>
          <p className="text-sm text-gray-600 mt-1">
            Saved on this purchase
          </p>
        </div>

        <div className="bg-gradient-to-br from-red-50 to-red-100 p-6 rounded-lg border-2 border-red-200">
          <div className="flex items-center justify-between mb-2">
            <Heart className="h-8 w-8 text-red-600" />
            <span className="text-sm font-medium text-red-700">Nutrition</span>
          </div>
          <p className="text-3xl font-bold text-red-600">
            +{totalNutrition}
          </p>
          <p className="text-sm text-gray-600 mt-1">
            HEI points improved
          </p>
        </div>

        <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-lg border-2 border-blue-200">
          <div className="flex items-center justify-between mb-2">
            <Clock className="h-8 w-8 text-blue-600" />
            <span className="text-sm font-medium text-blue-700">Speed</span>
          </div>
          <p className="text-3xl font-bold text-blue-600">
            {avgLatency}ms
          </p>
          <p className="text-sm text-gray-600 mt-1">
            Processing time
          </p>
        </div>
      </div>

      {/* Accepted Recommendations Summary */}
      {acceptedRecs.length > 0 && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            Accepted Recommendations
          </h3>
          <div className="space-y-2">
            {acceptedRecs.map((rec, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex-1">
                  <p className="font-medium text-gray-900">{rec.suggested_product}</p>
                  <p className="text-sm text-gray-500">
                    Replaced: {rec.original_product}
                  </p>
                </div>
                <div className="flex items-center space-x-4">
                  <span className="text-green-600 font-semibold">
                    -${rec.savings.toFixed(2)}
                  </span>
                  <span className="text-red-600 font-semibold">
                    +{rec.nutrition_improvement} HEI
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Insights */}
      <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-6 mb-6">
        <h3 className="text-lg font-semibold text-blue-900 mb-3">
          💡 Insights
        </h3>
        <ul className="space-y-2 text-gray-700">
          {totalSavings > 0 && (
            <li className="flex items-start">
              <span className="mr-2">•</span>
              <span>
                Your savings of ${totalSavings.toFixed(2)} could buy {Math.floor(totalSavings / 2)} more healthy items!
              </span>
            </li>
          )}
          {totalNutrition > 0 && (
            <li className="flex items-start">
              <span className="mr-2">•</span>
              <span>
                Improving your HEI score by {totalNutrition} points reduces chronic disease risk.
              </span>
            </li>
          )}
          <li className="flex items-start">
            <span className="mr-2">•</span>
            <span>
              The EAC system processed your cart in just {avgLatency}ms - faster than you can blink!
            </span>
          </li>
        </ul>
      </div>

      {/* Actions */}
      <div className="flex space-x-4">
        <button
          onClick={onReset}
          className="flex-1 bg-primary hover:bg-green-600 text-white font-semibold py-4 px-6 rounded-lg shadow-lg hover:shadow-xl transition flex items-center justify-center space-x-2"
        >
          <RotateCcw className="h-5 w-5" />
          <span>Start New Cart</span>
        </button>
      </div>
    </div>
  )
}
