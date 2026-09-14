import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || ''

const PRESETS = [
  { id: 'accra', name: 'Accra', label: '🌊 Coastal Urban', file: 'accra_thermal.tif' },
  { id: 'adilabad', name: 'Adilabad', label: '🌲 River Basin', file: 'adilabad_thermal.tif' },
  { id: 'abuja', name: 'Abuja', label: '🏙️ Urban Plateau', file: 'abuja_thermal.tif' },
]

function App() {
  const [file, setFile] = useState(null)
  const [inputPreview, setInputPreview] = useState(null)
  const [thermalPreview, setThermalPreview] = useState(null)
  const [outputImage, setOutputImage] = useState(null)
  const [displayImage, setDisplayImage] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingMessage, setLoadingMessage] = useState('')
  const [useTTA, setUseTTA] = useState(false)
  const [inferenceTime, setInferenceTime] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState(null)
  const [sliderPos, setSliderPos] = useState(50)
  const [claheApplied, setClaheApplied] = useState(false)
  const [apiStatus, setApiStatus] = useState('checking')
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'dark')
  const fileInputRef = useRef(null)
  const sliderContainerRef = useRef(null)
  const isDraggingSlider = useRef(false)

  // Track URLs for cleanup
  const blobUrlsRef = useRef([])
  const trackBlobUrl = useCallback((url) => {
    if (url && url.startsWith('blob:')) {
      blobUrlsRef.current.push(url)
    }
    return url
  }, [])

  const cleanupBlobUrls = useCallback(() => {
    blobUrlsRef.current.forEach(url => {
      try { URL.revokeObjectURL(url) } catch { /* ignore */ }
    })
    blobUrlsRef.current = []
  }, [])

  // Cleanup all blob URLs on unmount
  useEffect(() => {
    return () => cleanupBlobUrls()
  }, [cleanupBlobUrls])

  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => setTheme(prev => prev === 'dark' ? 'light' : 'dark')

  // Check API health on mount
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(res => res.ok ? setApiStatus('online') : setApiStatus('offline'))
      .catch(() => setApiStatus('offline'))
  }, [])

  const handleFile = useCallback((selectedFile) => {
    cleanupBlobUrls()
    setFile(selectedFile)
    setOutputImage(null)
    setDisplayImage(null)
    setInferenceTime(null)
    setError(null)
    setThermalPreview(null)
    setClaheApplied(false)
    setSliderPos(50)

    const isNpy = selectedFile.name.toLowerCase().endsWith('.npy')
    if (!isNpy) {
      const reader = new FileReader()
      reader.onload = (e) => setInputPreview(e.target.result)
      reader.readAsDataURL(selectedFile)
    } else {
      setInputPreview(null)
    }

    // Fetch thermal preview
    const formData = new FormData()
    formData.append('file', selectedFile)
    fetch(`${API_URL}/thermal-preview`, { method: 'POST', body: formData })
      .then(res => {
        if (!res.ok) throw new Error('Preview failed')
        return res.blob()
      })
      .then(blob => {
        const url = trackBlobUrl(URL.createObjectURL(blob))
        setThermalPreview(url)
      })
      .catch(() => { /* thermal preview is optional */ })
  }, [cleanupBlobUrls, trackBlobUrl])

  const loadPreset = async (preset) => {
    try {
      setLoading(true)
      setLoadingMessage(`Loading ${preset.name} preset…`)
      const res = await fetch(`/samples/${preset.file}`)
      if (!res.ok) throw new Error(`Could not load ${preset.file}`)
      const blob = await res.blob()
      const sampleFile = new File([blob], preset.file, { type: 'image/tiff' })
      handleFile(sampleFile)
    } catch (err) {
      setError(`Failed to load preset: ${err.message}`)
    } finally {
      setLoading(false)
      setLoadingMessage('')
    }
  }

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
    setLoadingMessage('Synthesizing optical RGB with Exp9 ResNet…')
    setError(null)
    setClaheApplied(false)

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
      const url = trackBlobUrl(URL.createObjectURL(blob))
      setOutputImage(url)
      setDisplayImage(url)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      setLoadingMessage('')
    }
  }

  const applyCLAHE = async () => {
    if (!outputImage) return

    setLoading(true)
    setLoadingMessage('Applying CLAHE enhancement…')

    try {
      const originalBlob = await fetch(outputImage).then(r => r.blob())
      const formData = new FormData()
      formData.append('file', originalBlob, 'output.png')

      const res = await fetch(`${API_URL}/postprocess/clahe`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) throw new Error('CLAHE failed')

      const blob = await res.blob()
      const url = trackBlobUrl(URL.createObjectURL(blob))
      setDisplayImage(url)
      setClaheApplied(true)
    } catch (err) {
      setError(`Post-processing failed: ${err.message}`)
    } finally {
      setLoading(false)
      setLoadingMessage('')
    }
  }

  const removeCLAHE = () => {
    setDisplayImage(outputImage)
    setClaheApplied(false)
  }

  const resetAll = () => {
    cleanupBlobUrls()
    setFile(null)
    setInputPreview(null)
    setThermalPreview(null)
    setOutputImage(null)
    setDisplayImage(null)
    setInferenceTime(null)
    setError(null)
    setClaheApplied(false)
    setSliderPos(50)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const downloadOutput = () => {
    if (!displayImage) return
    const a = document.createElement('a')
    a.href = displayImage
    a.download = `infranova_colorized_${claheApplied ? 'clahe_' : ''}${Date.now()}.png`
    a.click()
  }

  // Slider interaction
  const updateSliderPos = useCallback((clientX) => {
    const container = sliderContainerRef.current
    if (!container) return
    const rect = container.getBoundingClientRect()
    const x = clientX - rect.left
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100))
    setSliderPos(pct)
  }, [])

  const onSliderMouseDown = useCallback((e) => {
    isDraggingSlider.current = true
    updateSliderPos(e.clientX)
  }, [updateSliderPos])

  const onSliderTouchStart = useCallback((e) => {
    if (e.touches && e.touches[0]) {
      isDraggingSlider.current = true
      updateSliderPos(e.touches[0].clientX)
    }
  }, [updateSliderPos])

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (isDraggingSlider.current) updateSliderPos(e.clientX)
    }
    const handleMouseUp = () => {
      isDraggingSlider.current = false
    }
    const handleTouchMove = (e) => {
      if (isDraggingSlider.current && e.touches && e.touches[0]) {
        updateSliderPos(e.touches[0].clientX)
      }
    }
    const handleTouchEnd = () => {
      isDraggingSlider.current = false
    }

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    window.addEventListener('touchmove', handleTouchMove, { passive: true })
    window.addEventListener('touchend', handleTouchEnd)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
      window.removeEventListener('touchmove', handleTouchMove)
      window.removeEventListener('touchend', handleTouchEnd)
    }
  }, [updateSliderPos])

  const thermalSrc = thermalPreview || inputPreview

  return (
    <div className="app-shell">
      {/* Loading overlay */}
      {loading && (
        <div className="loading-overlay">
          <div className="loader-card">
            <div className="loader-ring" />
            <p className="loader-text">{loadingMessage || 'Processing…'}</p>
          </div>
        </div>
      )}

      {/* Navbar */}
      <nav className="navbar">
        <div className="container navbar__inner">
          <div className="navbar__brand">
            <span className="navbar__title">InfraNova AI</span>
            <span className="navbar__tag">v3.0</span>
            <span className={`status-dot status-dot--${apiStatus}`} title={`API ${apiStatus}`} />
          </div>
          <ul className="navbar__links">
            <li className="navbar__link" onClick={toggleTheme} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
              {theme === 'dark' ? '☀️' : '🌙'}
            </li>
            <li className="navbar__link" onClick={() => window.open('https://github.com/Soham-1009/InfraNova-AI', '_blank')}>
              GitHub
            </li>
          </ul>
        </div>
      </nav>

      {/* Main content */}
      <main className="main-content">
        {!file ? (
          /* ---- UPLOAD & LANDING STATE ---- */
          <div className="upload-view">
            <div className="hero fade-in-up">
              <div className="hero__badge">
                <span className="dot" />
                Exp9 Global ResNet · Landsat-9 Dual-Band (B10+B11) · Active Production
              </div>
              <h1 className="hero__title">
                <span className="gradient-thermal">Thermal</span> to <span className="gradient-rgb">True Color</span>
              </h1>
              <p className="hero__subtitle">
                Transform raw thermal infrared satellite rasters into photorealistic optical RGB imagery in real-time.
              </p>
            </div>

            <div
              id="upload-zone"
              className={`upload-zone fade-in-up fade-in-up--delay-2 ${dragOver ? 'dragover' : ''}`}
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

            {/* Presets Section */}
            <div className="presets-section fade-in-up fade-in-up--delay-3">
              <span className="presets-label">⚡ Quick Presets:</span>
              <div className="presets-grid">
                {PRESETS.map(p => (
                  <button
                    key={p.id}
                    className="preset-btn"
                    onClick={(e) => { e.stopPropagation(); loadPreset(p); }}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Model Architecture & Benchmarks Section */}
            <section className="architecture-section fade-in-up fade-in-up--delay-4">
              <h2 className="section-title">Production Model Architecture & Performance</h2>
              <div className="metrics-grid">
                <div className="metric-card">
                  <span className="metric-card__value">13.75 dB</span>
                  <span className="metric-card__label">Test Peak PSNR</span>
                </div>
                <div className="metric-card">
                  <span className="metric-card__value">0.450</span>
                  <span className="metric-card__label">Windowed SSIM</span>
                </div>
                <div className="metric-card">
                  <span className="metric-card__value">0.197 rad</span>
                  <span className="metric-card__label">Spectral Angle (SAM)</span>
                </div>
                <div className="metric-card">
                  <span className="metric-card__value">11.37M</span>
                  <span className="metric-card__label">Total Parameters</span>
                </div>
              </div>
            </section>
          </div>
        ) : (
          /* ---- WORKSPACE STATE ---- */
          <div className="workspace fade-in-up">
            {/* Top bar: file info + actions */}
            <div className="workspace__toolbar">
              <div className="toolbar__left">
                <span className="meta-chip">📄 <span className="value">{file.name}</span></span>
                <span className="meta-chip">📐 <span className="value">{(file.size / 1024).toFixed(1)} KB</span></span>
                {inferenceTime && (
                  <span className="meta-chip">⚡ <span className="value">{inferenceTime.toFixed(3)}s</span></span>
                )}
              </div>
              <div className="toolbar__right">
                <label className="toggle" title="Test-Time Augmentation: averages 4 geometric transforms for better quality">
                  <input
                    type="checkbox"
                    checked={useTTA}
                    onChange={(e) => setUseTTA(e.target.checked)}
                    id="tta-toggle"
                  />
                  <span className="toggle__label">TTA</span>
                </label>
                <button id="colorize-btn" className="btn btn--primary" onClick={runColorization} disabled={loading}>
                  ✨ Colorize
                </button>
                {displayImage && (
                  <>
                    {!claheApplied ? (
                      <button id="clahe-btn" className="btn btn--accent" onClick={applyCLAHE} disabled={loading} title="CLAHE: Contrast Limited Adaptive Histogram Equalization">
                        🔆 CLAHE
                      </button>
                    ) : (
                      <button id="clahe-undo-btn" className="btn btn--accent btn--active" onClick={removeCLAHE} disabled={loading}>
                        🔆 Undo CLAHE
                      </button>
                    )}
                    <button id="download-btn" className="btn btn--secondary" onClick={downloadOutput}>
                      ⬇ Download
                    </button>
                  </>
                )}
                <button id="reset-btn" className="btn btn--ghost" onClick={resetAll}>
                  ✕
                </button>
              </div>
            </div>

            {/* Error toast */}
            {error && (
              <div className="error-toast fade-in-up">
                <span className="error-toast__icon">⚠</span>
                <span className="error-toast__msg">{error}</span>
                <button className="error-toast__close" onClick={() => setError(null)}>✕</button>
              </div>
            )}

            {/* Image viewport */}
            <div className="viewport">
              {displayImage && thermalSrc ? (
                /* Interactive comparison slider */
                <div
                  className="slider-container"
                  ref={sliderContainerRef}
                  onMouseDown={onSliderMouseDown}
                  onTouchStart={onSliderTouchStart}
                >
                  {/* Output RGB (full bottom image) */}
                  <img className="slider-img slider-img--output" src={displayImage} alt="Generated RGB" draggable={false} />

                  {/* Thermal input (top image clipped at sliderPos) */}
                  <img
                    className="slider-img slider-img--input"
                    src={thermalSrc}
                    alt="Thermal input"
                    style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
                    draggable={false}
                  />

                  {/* Divider line */}
                  <div className="slider-divider" style={{ left: `${sliderPos}%` }}>
                    <div className="slider-handle">
                      <span className="slider-handle__arrow">◂</span>
                      <span className="slider-handle__arrow">▸</span>
                    </div>
                  </div>

                  {/* Labels */}
                  <span className="slider-label slider-label--left">Thermal IR (B10/B11)</span>
                  <span className="slider-label slider-label--right">
                    RGB Output{claheApplied ? ' + CLAHE' : ''}
                  </span>
                </div>
              ) : thermalSrc ? (
                /* Thermal-only preview */
                <div className="single-preview">
                  <img className="single-preview__img" src={thermalSrc} alt="Thermal preview" />
                  <span className="single-preview__label">Thermal IR Preview</span>
                  <p className="single-preview__hint">Click <strong>✨ Colorize</strong> to synthesize optical RGB</p>
                </div>
              ) : (
                /* Loading preview placeholder */
                <div className="single-preview single-preview--loading">
                  <div className="loader-ring loader-ring--small" />
                  <p className="single-preview__hint">Generating thermal preview…</p>
                </div>
              )}
            </div>

            {/* Stats bar */}
            {displayImage && (
              <div className="stats-bar fade-in-up">
                <div className="stat-pill"><span className="stat-pill__label">Resolution</span><span className="stat-pill__value">128×128</span></div>
                <div className="stat-pill"><span className="stat-pill__label">Inference</span><span className="stat-pill__value">{inferenceTime ? `${inferenceTime.toFixed(3)}s` : '—'}</span></div>
                <div className="stat-pill"><span className="stat-pill__label">Architecture</span><span className="stat-pill__value">Exp9 ResNet</span></div>
                <div className="stat-pill"><span className="stat-pill__label">TTA</span><span className="stat-pill__value">{useTTA ? 'On' : 'Off'}</span></div>
                <div className="stat-pill"><span className="stat-pill__label">CLAHE</span><span className="stat-pill__value">{claheApplied ? 'On' : 'Off'}</span></div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
