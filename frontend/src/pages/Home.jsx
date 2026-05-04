import { useState } from 'react'
import '../App.css'

export default function Home({ onResult }) {
  const [name, setName] = useState('')
  const [age, setAge] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const res = await fetch('/api/result', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim(), age: Number(age) }),
      })
      const data = await res.json()
      if (!res.ok) {
        setError(data.error || 'Something went wrong.')
      } else {
        onResult(data)
      }
    } catch {
      setError('Could not reach the server. Make sure the Flask backend is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <h1>Get Recommendations</h1>
      <p className="subtitle">Enter your name and age to receive personalised movie picks for your segment</p>

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="name">Name</label>
          <input
            id="name"
            type="text"
            placeholder="e.g. Ben Davis"
            value={name}
            onChange={e => setName(e.target.value)}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="age">Age</label>
          <input
            id="age"
            type="number"
            min="1"
            max="120"
            placeholder="e.g. 25"
            value={age}
            onChange={e => setAge(e.target.value)}
            required
          />
        </div>

        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? 'Loading…' : 'Get Recommendations'}
        </button>
      </form>

      {error && <div className="error-msg">{error}</div>}
    </div>
  )
}
