import '../App.css'

export default function Result({ data, onBack }) {
  const { segment, profile, recommendations } = data

  return (
    <div className="card result-card">
      <div className="result-header">
        <h1>Your Profile</h1>
        <span className="segment-badge">{segment}</span>
      </div>

      <div className="info-grid">
        <div className="info-item" style={{ gridColumn: '1 / -1' }}>
          <div className="label">Profile</div>
          <div className="value">{profile}</div>
        </div>
      </div>

      <div className="recs-section">
        <h2>Top Movie Recommendations</h2>
        <ul className="recs-list">
          {recommendations.map((title, i) => (
            <li key={i}>
              <span className="num">{i + 1}</span>
              {title}
            </li>
          ))}
        </ul>
      </div>

      <button className="btn-back" onClick={onBack}>
        ← Back
      </button>
    </div>
  )
}
