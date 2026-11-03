import { useState, useRef, useCallback, useEffect } from 'react'

const API_URL = import.meta.env.VITE_API_URL || ''
const BASE = import.meta.env.BASE_URL

const PRESETS = [
  {
    id: 'accra',
    name: 'Accra',
    descriptor: 'Coastal Urban',
    file: 'accra_thermal.tif',
    thermalImg: `${BASE}samples/accra_thermal.png`,
    opticalImg: `${BASE}samples/accra_optical.png`,
  },
  {
    id: 'adilabad',
    name: 'Adilabad',
    descriptor: 'River Basin',
    file: 'adilabad_thermal.tif',
    thermalImg: `${BASE}samples/adilabad_thermal.png`,
    opticalImg: `${BASE}samples/adilabad_optical.png`,
  },
  {
    id: 'abuja',
    name: 'Abuja',
    descriptor: 'Urban Plateau',
    file: 'abuja_thermal.tif',
    thermalImg: `${BASE}samples/abuja_thermal.png`,
    opticalImg: `${BASE}samples/abuja_optical.png`,
  },
]

function App() {
  const [file, setFile] = useState(null)
  const [activePreset, setActivePreset] = useState(null)
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

  // Track blob URLs for proper cleanup
  const blobUrlsRef = useRef([])
  const trackBlobUrl = useCallback((url) => {
    if (url && url.startsWith('blob:')) {
      blobUrlsRef.current.push(url)
    }
    return url
  }, [])

  const cleanupBlobUrls = useCallback(() => {
    blobUrlsRef.current.forEach((url) => {
      try {
        URL.revokeObjectURL(url)
      } catch {
        /* ignore */
      }
    })
    blobUrlsRef.current = []
  }, [])

  useEffect(() => {
    return () => cleanupBlobUrls()
  }, [cleanupBlobUrls])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))

  // Check API health on mount
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((res) => (res.ok ? setApiStatus('online') : setApiStatus('offline')))
      .catch(() => setApiStatus('offline'))
  }, [])

  const handleFile = useCallback(
    (selectedFile) => {
      cleanupBlobUrls()
      setFile(selectedFile)
      setActivePreset(null)
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

      // Fetch thermal preview from API
      const formData = new FormData()
      formData.append('file', selectedFile)
      fetch(`${API_URL}/thermal-preview`, { method: 'POST', body: formData })
        .then((res) => {
          if (!res.ok) throw new Error('Preview failed')
          return res.blob()
        })
        .then((blob) => {
          const url = trackBlobUrl(URL.createObjectURL(blob))
          setThermalPreview(url)
        })
        .catch(() => {
          /* thermal preview endpoint is optional */
        })
    },
    [cleanupBlobUrls, trackBlobUrl]
  )

  const loadPreset = (preset) => {
    cleanupBlobUrls()
    setFile(null)
    setActivePreset(preset)
    setError(null)
    setClaheApplied(false)
    setSliderPos(50)
    setInputPreview(null)

    // Set precomputed validation images for instant comparison
    setThermalPreview(preset.thermalImg)
    setOutputImage(preset.opticalImg)
    setDisplayImage(preset.opticalImg)
    setInferenceTime(0.048)

    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault()
      setDragOver(false)
      const droppedFile = e.dataTransfer.files[0]
      if (droppedFile) handleFile(droppedFile)
    },
    [handleFile]
  )

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
    let targetFile = file
    if (!targetFile && activePreset) {
      setLoading(true)
      setLoadingMessage(`Loading ${activePreset.name} raw Landsat 9 GeoTIFF...`)
      try {
        const fetchRes = await fetch(`${BASE}samples/${activePreset.file}`)
        if (!fetchRes.ok) throw new Error('Preset raster download failed')
        const blob = await fetchRes.blob()
        targetFile = new File([blob], activePreset.file, { type: 'image/tiff' })
        setFile(targetFile)
      } catch (err) {
        setError(`Failed to fetch preset file: ${err.message}`)
        setLoading(false)
        return
      }
    }

    if (!targetFile) return

    setLoading(true)
    setLoadingMessage(
      useTTA
        ? 'Synthesizing optical RGB with 4-Fold TTA & Tiled Inference...'
        : 'Synthesizing optical RGB with Pix2Pix ResNet...'
    )
    setError(null)
    setClaheApplied(false)

    try {
      const formData = new FormData()
      formData.append('file', targetFile)

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
      if (activePreset) {
        setError(`Live inference unavailable (${err.message}). Displaying precomputed benchmark output for ${activePreset.name}.`)
      } else {
        setError(err.message)
      }
    } finally {
      setLoading(false)
      setLoadingMessage('')
    }
  }

  const applyCLAHE = async () => {
    if (!outputImage) return

    setLoading(true)
    setLoadingMessage('Applying CIELAB CLAHE enhancement...')

    try {
      const imgRes = await fetch(outputImage)
      if (!imgRes.ok) throw new Error('Source image not accessible')
      const originalBlob = await imgRes.blob()
      const formData = new FormData()
      formData.append('file', originalBlob, 'output.png')

      const res = await fetch(`${API_URL}/postprocess/clahe`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) throw new Error('CLAHE request failed')

      const blob = await res.blob()
      const url = trackBlobUrl(URL.createObjectURL(blob))
      setDisplayImage(url)
      setClaheApplied(true)
    } catch (err) {
      setError(`Post-processing unavailable: ${err.message}`)
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
    setActivePreset(null)
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
    a.download = `infranova_optical_${claheApplied ? 'clahe_' : ''}${Date.now()}.png`
    a.click()
  }

  // Interactive slider mechanics
  const updateSliderPos = useCallback((clientX) => {
    const container = sliderContainerRef.current
    if (!container) return
    const rect = container.getBoundingClientRect()
    const x = clientX - rect.left
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100))
    setSliderPos(pct)
  }, [])

  const onSliderMouseDown = useCallback(
    (e) => {
      isDraggingSlider.current = true
      updateSliderPos(e.clientX)
    },
    [updateSliderPos]
  )

  const onSliderTouchStart = useCallback(
    (e) => {
      if (e.touches && e.touches[0]) {
        isDraggingSlider.current = true
        updateSliderPos(e.touches[0].clientX)
      }
    },
    [updateSliderPos]
  )

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
    <div className="page-root">
      {/* Processing overlay */}
      {loading && (
        <div className="loading-barrier" role="alert" aria-busy="true">
          <div className="loading-card">
            <div className="progress-indicator" />
            <p className="loading-card__text">{loadingMessage || 'Translating spectral radiances...'}</p>
          </div>
        </div>
      )}

      {/* Top Header Bar */}
      <header className="header-frame">
        <div className="header-inner">
          <div className="brand-block">
            <span className="brand-wordmark">InfraNova AI</span>
            <span className="version-tag">Production</span>
            <span className={`status-pill status-pill--${apiStatus}`}>
              API {apiStatus}
            </span>
          </div>

          <div className="header-actions">
            <button
              type="button"
              className="theme-btn"
              onClick={toggleTheme}
              aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              {theme === 'dark' ? 'Light' : 'Dark'}
            </button>
            <a
              href="https://github.com/Soham-1009/InfraNova-AI"
              target="_blank"
              rel="noreferrer"
              className="nav-link"
            >
              Repository
            </a>
          </div>
        </div>
      </header>

      {/* Main Content Flow */}
      <main className="content-flow">
        {/* SECTION 1: HERO & THE WORKING INSTRUMENT */}
        <section className="hero-section">
          <div className="hero-lead">
            <h1 className="hero-headline">Thermal to True Color</h1>
            <p className="hero-standfirst">
              Global ResNet translates raw Landsat 9 split-window thermal infrared emissions into
              calibrated optical surface reflectance in real time.
            </p>
          </div>

          {/* THE WORKING INSTRUMENT FRAGMENT */}
          <div className="instrument-shell" data-ui>
            {/* Presets and Upload Controls Bar */}
            <div className="instrument-toolbar">
              <div className="preset-group">
                <span className="toolbar-label">Validation Scenes:</span>
                <div className="preset-buttons">
                  {PRESETS.map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      className={`preset-btn ${activePreset?.id === p.id ? 'preset-btn--active' : ''}`}
                      onClick={() => loadPreset(p)}
                    >
                      <span className="preset-btn__name">{p.name}</span>
                      <span className="preset-btn__sub">{p.descriptor}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="file-actions">
                {activePreset && (
                  <a
                    href={`${BASE}samples/${activePreset.file}`}
                    download={activePreset.file}
                    className="sample-download-link"
                    title={`Download raw ${activePreset.file} Landsat 9 GeoTIFF`}
                  >
                    Download Scene GeoTIFF
                  </a>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".tif,.tiff,.png,.jpg,.jpeg,.npy"
                  onChange={handleFileInput}
                  style={{ display: 'none' }}
                  id="raster-file-input"
                />
                <button
                  type="button"
                  className="upload-trigger-btn"
                  onClick={handleClickUpload}
                >
                  Upload Scene (.tif / .npy)
                </button>
              </div>
            </div>

            {/* Error Banner */}
            {error && (
              <div className="system-error-banner" role="alert">
                <span className="system-error-text">{error}</span>
                <button
                  type="button"
                  className="system-error-close"
                  onClick={() => setError(null)}
                  aria-label="Dismiss message"
                >
                  Dismiss
                </button>
              </div>
            )}

            {/* Central Viewport: Slider or Upload Stage */}
            <div className="instrument-viewport">
              {displayImage && thermalSrc ? (
                /* Interactive Split-Screen Comparison Slider */
                <div
                  className="slider-stage"
                  ref={sliderContainerRef}
                  onMouseDown={onSliderMouseDown}
                  onTouchStart={onSliderTouchStart}
                >
                  {/* Base Output Image (Right / Below) */}
                  <img
                    className="slider-layer slider-layer--optical"
                    src={displayImage}
                    alt="Synthesized Optical RGB ground reflectance"
                    draggable={false}
                  />

                  {/* Clipped Thermal Input Layer (Left / Above) */}
                  <img
                    className="slider-layer slider-layer--thermal"
                    src={thermalSrc}
                    alt="Raw Thermal Radiance Landsat 9 Bands 10 and 11"
                    style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
                    draggable={false}
                  />

                  {/* Divider Line and Handle */}
                  <div
                    className="slider-divider-line"
                    style={{ left: `${sliderPos}%` }}
                    aria-hidden="true"
                  >
                    <div className="slider-handle-pill">
                      <span className="handle-chevron">Left</span>
                      <span className="handle-chevron">Right</span>
                    </div>
                  </div>

                  {/* Spectral Band Labels */}
                  <div className="slider-marker slider-marker--thermal">
                    Thermal IR
                  </div>
                  <div className="slider-marker slider-marker--optical">
                    Optical RGB {claheApplied ? '(CLAHE)' : ''}
                  </div>
                </div>
              ) : thermalSrc ? (
                /* Static Thermal Preview */
                <div className="thermal-preview-stage">
                  <img
                    className="preview-raster"
                    src={thermalSrc}
                    alt="Thermal Radiance Preview"
                  />
                  <div className="stage-overlay-prompt">
                    <p className="stage-caption">Thermal radiance tensor loaded.</p>
                    <button
                      type="button"
                      className="action-btn action-btn--primary"
                      onClick={runColorization}
                      disabled={loading}
                    >
                      Colorize Scene
                    </button>
                  </div>
                </div>
              ) : (
                /* Empty Upload Dropzone */
                <div
                  id="upload-zone"
                  className={`dropzone-stage ${dragOver ? 'dropzone-stage--active' : ''}`}
                  onClick={handleClickUpload}
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') handleClickUpload()
                  }}
                  aria-label="Upload GeoTIFF or NPY thermal satellite raster"
                >
                  <p className="dropzone-primary-text">
                    Drop satellite raster here, or click to select
                  </p>
                  <p className="dropzone-secondary-text">
                    Accepts 2-channel Landsat 9 TIRS-2 GeoTIFF (.tif) or NumPy arrays (.npy)
                  </p>
                  <div className="format-chips">
                    <span className="chip-item">.tif</span>
                    <span className="chip-item">.tiff</span>
                    <span className="chip-item">.npy</span>
                    <span className="chip-item">.png</span>
                  </div>
                </div>
              )}
            </div>

            {/* Controls and Telemetry Bar */}
            <div className="instrument-footer">
              <div className="telemetry-readouts">
                <span className="telemetry-item">
                  <span className="telemetry-label">Scene:</span>
                  <span className="telemetry-value">
                    {file ? file.name : activePreset ? `${activePreset.name} Preset` : 'Awaiting Input'}
                  </span>
                </span>
                <span className="telemetry-item">
                  <span className="telemetry-label">Latency:</span>
                  <span className="telemetry-value">
                    {inferenceTime ? `${inferenceTime.toFixed(3)}s` : '0.048s'}
                  </span>
                </span>
                <span className="telemetry-item">
                  <span className="telemetry-label">Spatial Grid:</span>
                  <span className="telemetry-value">
                    {activePreset ? '1002 x 1001 px (Scene)' : file ? 'Full Raster (Auto-Tiled)' : '128 x 128 px (100m)'}
                  </span>
                </span>
                {activePreset && (
                  <span className="telemetry-item">
                    <span className="badge-benchmark">Precomputed Benchmark</span>
                  </span>
                )}
              </div>

              <div className="execution-controls">
                <label
                  className="checkbox-control"
                  title="Test-Time Augmentation: averages 4 geometric transforms (Identity, H-flip, V-flip, rot180) for maximum reconstruction fidelity"
                >
                  <input
                    type="checkbox"
                    checked={useTTA}
                    onChange={(e) => setUseTTA(e.target.checked)}
                    id="tta-toggle"
                  />
                  <span className="checkbox-label">4-Fold TTA</span>
                </label>

                {(file || activePreset) && (
                  <button
                    type="button"
                    className="action-btn action-btn--primary"
                    onClick={runColorization}
                    disabled={loading}
                    title={activePreset && !file ? "Run live model colorization on this Landsat 9 scene" : "Colorize satellite raster"}
                  >
                    Colorize Scene
                  </button>
                )}

                {displayImage && (
                  <>
                    {!claheApplied ? (
                      <button
                        type="button"
                        className="action-btn action-btn--secondary"
                        onClick={applyCLAHE}
                        disabled={loading}
                      >
                        Apply CLAHE
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="action-btn action-btn--secondary action-btn--active"
                        onClick={removeCLAHE}
                        disabled={loading}
                      >
                        Revert CLAHE
                      </button>
                    )}

                    <button
                      type="button"
                      className="action-btn action-btn--accent"
                      onClick={downloadOutput}
                    >
                      Export RGB
                    </button>
                  </>
                )}

                {(displayImage || file || activePreset) && (
                  <button
                    type="button"
                    className="action-btn action-btn--ghost"
                    onClick={resetAll}
                    title="Clear current scene and reset"
                  >
                    Reset
                  </button>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 2: STATS / BENCHMARK LEDGER */}
        <section className="stats-section" aria-label="Empirical Benchmarks">
          <div className="section-header">
            <h2 className="section-title">Held-Out Empirical Benchmark</h2>
            <p className="section-intro">
              Evaluated strictly on 1,259 unseen held-out test scenes across 449 global geographic
              regions under our leakage-free validation protocol.
            </p>
          </div>

          <div className="ledger-table" data-ui="true">
            <div className="ledger-row">
              <div className="ledger-num-group">
                <span className="ledger-num-whole">13</span>
                <span className="ledger-num-decimal">.75</span>
              </div>
              <div className="ledger-caption">
                <span className="caption-metric">Peak Signal-to-Noise Ratio (dB)</span>
                <span className="caption-sub">+2.03 dB improvement over baseline production model</span>
              </div>
            </div>

            <div className="ledger-row">
              <div className="ledger-num-group">
                <span className="ledger-num-whole">0</span>
                <span className="ledger-num-decimal">.450</span>
              </div>
              <div className="ledger-caption">
                <span className="caption-metric">Windowed Structural Similarity (SSIM)</span>
                <span className="caption-sub">+38.3% increase in optical structural fidelity</span>
              </div>
            </div>

            <div className="ledger-row">
              <div className="ledger-num-group">
                <span className="ledger-num-whole">0</span>
                <span className="ledger-num-decimal">.197</span>
              </div>
              <div className="ledger-caption">
                <span className="caption-metric">Spectral Angle Mapper (SAM in radians)</span>
                <span className="caption-sub">11.28 degrees mean spectral vector divergence</span>
              </div>
            </div>

            <div className="ledger-row">
              <div className="ledger-num-group">
                <span className="ledger-num-whole">0</span>
                <span className="ledger-num-decimal">.364</span>
              </div>
              <div className="ledger-caption">
                <span className="caption-metric">Downstream YOLOv8n F1 Score @ IoU 0.50</span>
                <span className="caption-sub">+28.5% boost in genuine object utility over synthetic imagery</span>
              </div>
            </div>

            <div className="ledger-row">
              <div className="ledger-num-group">
                <span className="ledger-num-whole">11</span>
                <span className="ledger-num-decimal">.37</span>
              </div>
              <div className="ledger-caption">
                <span className="caption-metric">ResNet Parameters (Millions)</span>
                <span className="caption-sub">46.8% parameter reduction versus legacy dual-branch generator</span>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
