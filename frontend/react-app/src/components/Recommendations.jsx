import { Sparkles, Check, X, TrendingUp, DollarSign, Heart } from 'lucide-react'
import { useState } from 'react'

export default function Recommendations({ recommendations, onAccept, onDecline, onComplete }) {
  const [processedRecs, setProcessedRecs] = useState(new Set())

  const handleAccept = (recId) => {
    setProcessedRecs(new Set([...processedRecs, recId]))
    onAccept(recId)
  }

  const handleDecline = (recId) => {
    setProcessedRecs(new Set([...processedRecs, recId]))
    onDecline(recId)
  }

  const allProcessed = recommendations.every(r => processedRecs.has(r.id))

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900 flex items-center">
          <Sparkles className="mr-2 h-6 w-6 text-primary" />
          Smart Recommendations
        </h2>
        <span className="text-sm text-gray-500">
          {recommendations.length} suggestions
        </span>
      </div>

      {recommendations.length === 0 ? (
        <div className="text-center py-12">
          <Sparkles className="mx-auto h-12 w-12 text-gray-400" />
          <p className="mt-4 text-gray-500">No recommendations available</p>
        </div>
      ) : (
        <>
          <div className="space-y-4 mb-6">
            {recommendations.map((rec) => (
              <div
                key={rec.id}
                className={`p-6 rounded-lg border-2 transition ${
                  processedRecs.has(rec.id)
                    ? 'border-gray-200 bg-gray-50 opacity-60'
                    : 'border-primary bg-green-50'
                }`}
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-sm font-medium text-gray-500">Replace:</span>
                      <span className="font-semibold text-gray-900">{rec.original_product}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-medium text-primary">With:</span>
                      <span className="font-bold text-primary text-lg">{rec.suggested_product}</span>
                    </div>
                  </div>
                  <div className="flex items-center space-x-1 px-3 py-1 bg-primary text-white rounded-full text-sm font-medium">
                    <TrendingUp className="h-4 w-4" />
                    <span>{Math.round(rec.confidence * 100)}%</span>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-4 mb-4">
                  <div className="flex items-center space-x-2 p-3 bg-white rounded-lg">
                    <DollarSign className="h-5 w-5 text-green-600" />
                    <div>
                      <p className="text-xs text-gray-500">Save</p>
                      <p className="font-bold text-green-600">${rec.savings.toFixed(2)}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2 p-3 bg-white rounded-lg">
                    <Heart className="h-5 w-5 text-red-600" />
                    <div>
                      <p className="text-xs text-gray-500">Nutrition</p>
                      <p className="font-bold text-red-600">+{rec.nutrition_improvement}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2 p-3 bg-white rounded-lg">
                    <TrendingUp className="h-5 w-5 text-blue-600" />
                    <div>
                      <p className="text-xs text-gray-500">HEI</p>
                      <p className="font-bold text-blue-600">Better</p>
                    </div>
                  </div>
                </div>

                <p className="text-sm text-gray-600 mb-4 italic">
                  💡 {rec.reason}
                </p>

                {!processedRecs.has(rec.id) && (
                  <div className="flex space-x-3">
                    <button
                      onClick={() => handleAccept(rec.id)}
                      className="flex-1 bg-primary hover:bg-green-600 text-white font-semibold py-3 px-4 rounded-lg shadow hover:shadow-lg transition flex items-center justify-center space-x-2"
                    >
                      <Check className="h-5 w-5" />
                      <span>Accept</span>
                    </button>
                    <button
                      onClick={() => handleDecline(rec.id)}
                      className="flex-1 bg-gray-200 hover:bg-gray-300 text-gray-700 font-semibold py-3 px-4 rounded-lg transition flex items-center justify-center space-x-2"
                    >
                      <X className="h-5 w-5" />
                      <span>Decline</span>
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>

          {allProcessed && (
            <button
              onClick={onComplete}
              className="w-full bg-secondary hover:bg-blue-600 text-white font-semibold py-4 px-6 rounded-lg shadow-lg hover:shadow-xl transition flex items-center justify-center space-x-2"
            >
              <Check className="h-5 w-5" />
              <span>Complete Purchase</span>
            </button>
          )}
        </>
      )}
    </div>
  )
}
