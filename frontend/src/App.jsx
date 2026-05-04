import { useState } from 'react'
import Home from './pages/Home'
import Result from './pages/Result'
import './App.css'

export default function App() {
  const [result, setResult] = useState(null)

  return (
    <div className="app-wrapper">
      <div className="logo-header">
        <span className="logo-icon">🎬</span>
        <span className="logo-text">CineMatch</span>
      </div>
      {result
        ? <Result data={result} onBack={() => setResult(null)} />
        : <Home onResult={setResult} />}
    </div>
  )
}
