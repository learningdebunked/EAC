import { ShoppingCart, Trash2, Plus } from 'lucide-react'

export default function Cart({ cart, setCart, onCheckout }) {
  const total = cart.reduce((sum, item) => sum + item.price, 0)

  const removeItem = (id) => {
    setCart(cart.filter(item => item.id !== id))
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900 flex items-center">
          <ShoppingCart className="mr-2 h-6 w-6 text-primary" />
          Your Cart
        </h2>
        <span className="text-sm text-gray-500">{cart.length} items</span>
      </div>

      {cart.length === 0 ? (
        <div className="text-center py-12">
          <ShoppingCart className="mx-auto h-12 w-12 text-gray-400" />
          <p className="mt-4 text-gray-500">Your cart is empty</p>
        </div>
      ) : (
        <>
          <div className="space-y-4 mb-6">
            {cart.map((item) => (
              <div key={item.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition">
                <div className="flex-1">
                  <h3 className="font-medium text-gray-900">{item.name}</h3>
                  <p className="text-sm text-gray-500">{item.category}</p>
                  <div className="mt-1 flex items-center space-x-2">
                    <span className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded">
                      HEI: {item.hei}
                    </span>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <span className="text-lg font-semibold text-gray-900">
                    ${item.price.toFixed(2)}
                  </span>
                  <button
                    onClick={() => removeItem(item.id)}
                    className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition"
                  >
                    <Trash2 className="h-5 w-5" />
                  </button>
                </div>
              </div>
            ))}
          </div>

          <div className="border-t pt-4">
            <div className="flex justify-between items-center mb-6">
              <span className="text-xl font-bold text-gray-900">Total</span>
              <span className="text-2xl font-bold text-primary">
                ${total.toFixed(2)}
              </span>
            </div>

            <button
              onClick={onCheckout}
              disabled={cart.length === 0}
              className="w-full bg-primary hover:bg-green-600 text-white font-semibold py-4 px-6 rounded-lg shadow-lg hover:shadow-xl transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              <ShoppingCart className="h-5 w-5" />
              <span>Proceed to Checkout</span>
            </button>
          </div>
        </>
      )}
    </div>
  )
}
