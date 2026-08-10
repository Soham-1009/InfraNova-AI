import { useState, useRef, useCallback } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [file, setFile] = useState(null)
  const [inputPreview, setInputPreview] = useState(null)
  const [thermalPreview, setThermalPreview] = useState(null)
  const [outputImage, setOutputImage] = useState(null)
  const [loading, setLoading] = useState(false)
  const [useTTA, setUseTTA] = useState(false)
  const [inferenceTime, setInferenceTime] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleFile = useCallback((selectedFile) => {
    setFile(selectedFile)
    setOutputImage(null)
    setInferenceTime(null)
    setError(null)
    setThermalPreview(null)

    // Create a local preview for the file
    const isNpy = selectedFile.name.toLowerCase().endsWith('.npy')
    if (!isNpy) {
      const reader = new FileReader()
      reader.onload = (e) => setInputPreview(e.target.result)
      reader.readAsDataURL(selectedFile)
    } else {
      setInputPreview(null)
    }

    // Fetch thermal preview from backend
    const formData = new FormData()
    formData.append('file', selectedFile)
    fetch(`${API_URL}/thermal-preview`, { method: 'POST', body: formData })
      .then(res => {
        if (!res.ok) throw new Error('Preview failed')
        return res.blob()
      })
      .then(blob => setThermalPreview(URL.createObjectURL(blob)))
      .catch(() => { /* thermal preview is optional */ })
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile) handleFile(droppedFile)
  }, [handleFile])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setDragOver(false)
  }, [])

  const handleClickUpload = () => {
    fileInputRef.current?.click()
  }

  const handleFileInput = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) handleFile(selectedFile)
  }

  const runColorization = async () => {
    if (!file) return

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch(`${API_URL}/colorize?tta=${useTTA}`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || 'Colorization failed')
      }

      const time = res.headers.get('X-Inference-Time')
      if (time) setInferenceTime(parseFloat(time))

      const blob = await res.blob()
      setOutputImage(URL.createObjectURL(blob))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const resetAll = () => {
    setFile(null)
    setInputPreview(null)
    setThermalPreview(null)
    setOutputImage(null)
    setInferenceTime(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const downloadOutput = () => {
    if (!outputImage) return
    const a = document.createElement('a')
    a.href = outputImage
    a.download = `infranova_colorized_${Date.now()}.png`
    a.click()
  }

  return (
    <>
      {/* Loading Overlay */}
      {loading && (
        <div className="loader-overlay">
          <div className="loader-ring" />
          <p className="loader-text">Colorizing satellite imagery…</p>
        </div>
      )}

      {/* Navbar */}
      <nav className="navbar">
        <div className="container">
          <div className="navbar__brand">
            <span className="navbar__title">InfraNova AI</span>
            <span className="navbar__tag">v1.0</span>
          </div>
          <ul className="navbar__links">
            <li className="navbar__link" onClick={() => window.open('https://github.com/Soham-1009/InfraNova-AI', '_blank')}>
              GitHub
            </li>
          </ul>
        </div>
      </nav>

      {/* Hero */}
      <section className="hero">
        <div className="container">
          <div className="hero__badge fade-in-up">
            <span className="dot" />
            Pix2Pix · Landsat-9 · Epoch 226
          </div>
          <h1 className="hero__title fade-in-up fade-in-up--delay-1">
            Thermal to <span className="gradient">True Color</span>
          </h1>
          <p className="hero__subtitle fade-in-up fade-in-up--delay-2">
            Upload a thermal infrared satellite image and watch our AI model 
            generate a photorealistic RGB colorization in seconds.
          </p>
        </div>
      </section>

      {/* Upload Section */}
      <section className="upload-section">
        <div className="container">
          {!file ? (
            <div
              id="upload-zone"
              className={`upload-zone fade-in-up fade-in-up--delay-3 ${dragOver ? 'dragover' : ''}`}
              onClick={handleClickUpload}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
            >
              <div className="upload-zone__icon">🛰️</div>
              <p className="upload-zone__title">Drop your thermal image here</p>
              <p className="upload-zone__subtitle">or click to browse files</p>
              <div className="upload-zone__formats">
                {['.tif', '.tiff', '.png', '.jpg', '.npy'].map(ext => (
                  <span key={ext} className="format-tag">{ext}</span>
                ))}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".tif,.tiff,.png,.jpg,.jpeg,.npy"
                onChange={handleFileInput}
                style={{ display: 'none' }}
                id="file-input"
              />
            </div>
          ) : (
            <>
              {/* File Info + Controls */}
              <div className="controls fade-in-up">
                <span className="meta-chip">
                  📄 <span className="value">{file.name}</span>
                </span>
                <span className="meta-chip">
                  📐 <span className="value">{(file.size / 1024).toFixed(1)} KB</span>
                </span>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={useTTA}
                    onChange={(e) => setUseTTA(e.target.checked)}
                    id="tta-toggle"
                  />
                  TTA
                </label>
                <button
                  id="colorize-btn"
                  className="btn btn--primary"
                  onClick={runColorization}
                  disabled={loading}
                >
                  ✨ Colorize
                </button>
                <button
                  id="reset-btn"
                  className="btn btn--secondary"
                  onClick={resetAll}
                >
                  ↻ Reset
                </button>
              </div>

              {/* Error */}
              {error && (
                <div style={{
                  marginTop: 'var(--space-md)',
                  padding: '12px 16px',
                  background: 'rgba(239, 68, 68, 0.1)',
                  border: '1px solid rgba(239, 68, 68, 0.25)',
                  borderRadius: 'var(--radius-md)',
                  color: 'var(--error)',
                  fontSize: '0.85rem',
                  textAlign: 'center',
                }}>
                  {error}
                </div>
              )}
            </>
          )}
        </div>
      </section>

      {/* Results Section */}
      {(thermalPreview || outputImage) && (
        <section className="results-section">
          <div className="container">
            <div className="results__header">
              <h2 className="results__title">
                {outputImage ? '🎨 Comparison' : '🌡️ Thermal Preview'}
              </h2>
              <div className="results__meta">
                {inferenceTime && (
                  <span className="meta-chip">
                    ⚡ <span className="value">{inferenceTime.toFixed(3)}s</span>
                  </span>
                )}
                {outputImage && (
                  <button id="download-btn" className="btn btn--secondary" onClick={downloadOutput}>
                    ⬇ Download
                  </button>
                )}
              </div>
            </div>

            <div className="comparison">
              {/* Input Panel */}
              <div className="comparison__panel fade-in-up">
                <span className="comparison__label comparison__label--input">
                  Thermal IR
                </span>
                <img
                  className="comparison__image"
                  src={thermalPreview || inputPreview}
                  alt="Thermal input"
                />
              </div>

              {/* Output Panel */}
              {outputImage && (
                <div className="comparison__panel fade-in-up fade-in-up--delay-1">
                  <span className="comparison__label comparison__label--output">
                    RGB Output
                  </span>
                  <img
                    className="comparison__image"
                    src={outputImage}
                    alt="Colorized output"
                  />
                </div>
              )}
            </div>

            {/* Stats */}
            {outputImage && (
              <div className="stats fade-in-up fade-in-up--delay-2">
                <div className="stat-card">
                  <div className="stat-card__value">128×128</div>
                  <div className="stat-card__label">Resolution</div>
                </div>
                <div className="stat-card">
                  <div className="stat-card__value">{inferenceTime ? `${inferenceTime.toFixed(2)}s` : '—'}</div>
                  <div className="stat-card__label">Inference Time</div>
                </div>
                <div className="stat-card">
                  <div className="stat-card__value">Epoch 226</div>
                  <div className="stat-card__label">Model Version</div>
                </div>
                <div className="stat-card">
                  <div className="stat-card__value">{useTTA ? 'On' : 'Off'}</div>
                  <div className="stat-card__label">TTA Mode</div>
                </div>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p className="footer__text">
            Powered by PyTorch & Pix2Pix GAN
          </p>
        </div>
      </footer>
    </>
  )
}

export default App
