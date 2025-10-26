import { User, DollarSign, Users, AlertCircle } from 'lucide-react'

export default function UserProfile({ profile, setProfile }) {
  const updateProfile = (key, value) => {
    setProfile({ ...profile, [key]: value })
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 sticky top-4">
      <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
        <User className="mr-2 h-5 w-5 text-primary" />
        User Profile
      </h2>

      <div className="space-y-4">
        {/* Income */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <DollarSign className="inline h-4 w-4 mr-1" />
            Annual Income
          </label>
          <input
            type="number"
            value={profile.income}
            onChange={(e) => updateProfile('income', parseInt(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
          />
        </div>

        {/* Household Size */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <Users className="inline h-4 w-4 mr-1" />
            Household Size
          </label>
          <input
            type="number"
            value={profile.household_size}
            onChange={(e) => updateProfile('household_size', parseInt(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
          />
        </div>

        {/* SNAP Eligible */}
        <div>
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={profile.snap_eligible}
              onChange={(e) => updateProfile('snap_eligible', e.target.checked)}
              className="w-4 h-4 text-primary border-gray-300 rounded focus:ring-primary"
            />
            <span className="text-sm font-medium text-gray-700">
              SNAP Eligible
            </span>
          </label>
        </div>

        {/* Food Insecurity */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <AlertCircle className="inline h-4 w-4 mr-1" />
            Food Insecurity Level
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={profile.food_insecurity}
            onChange={(e) => updateProfile('food_insecurity', parseFloat(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Low</span>
            <span className="font-semibold">{(profile.food_insecurity * 100).toFixed(0)}%</span>
            <span>High</span>
          </div>
        </div>

        {/* Diabetes Risk */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Diabetes Risk
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={profile.diabetes_risk}
            onChange={(e) => updateProfile('diabetes_risk', parseFloat(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Low</span>
            <span className="font-semibold">{(profile.diabetes_risk * 100).toFixed(0)}%</span>
            <span>High</span>
          </div>
        </div>
      </div>

      {/* Profile Summary */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Profile Summary</h3>
        <div className="space-y-1 text-xs text-gray-600">
          <p>Income: ${profile.income.toLocaleString()}/year</p>
          <p>Household: {profile.household_size} people</p>
          <p>SNAP: {profile.snap_eligible ? 'Eligible' : 'Not eligible'}</p>
          <p>Food Insecurity: {(profile.food_insecurity * 100).toFixed(0)}%</p>
          <p>Diabetes Risk: {(profile.diabetes_risk * 100).toFixed(0)}%</p>
        </div>
      </div>
    </div>
  )
}
