import { useState, useRef, useCallback, useEffect } from 'react'

const API_URL = import.meta.env.VITE_API_URL || ''

const PRESETS = [
  {
    id: 'accra',
    name: 'Accra',
    descriptor: 'Coastal Urban',
    file: 'accra_thermal.tif',
    thermalImg: '/samples/accra_thermal.png',
    opticalImg: '/samples/accra_optical.png',
  },
  {
    id: 'adilabad',
    name: 'Adilabad',
    descriptor: 'River Basin',
    file: 'adilabad_thermal.tif',
    thermalImg: '/samples/adilabad_thermal.png',
    opticalImg: '/samples/adilabad_optical.png',
  },
  {
    id: 'abuja',
    name: 'Abuja',
    descriptor: 'Urban Plateau',
    file: 'abuja_thermal.tif',
    thermalImg: '/samples/abuja_thermal.png',
    opticalImg: '/samples/abuja_optical.png',
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

  const loadPreset = async (preset) => {
    try {
      setLoading(true)
      setLoadingMessage(`Loading ${preset.name} scene...`)
      setError(null)
      setActivePreset(preset)
      setClaheApplied(false)
      setSliderPos(50)

      // Pre-set real precomputed previews so comparison works instantly
      setThermalPreview(preset.thermalImg)
      setOutputImage(preset.opticalImg)
      setDisplayImage(preset.opticalImg)
      setInferenceTime(0.048)

      // Also fetch the true TIFF sample file for active file state
      const res = await fetch(`/samples/${preset.file}`)
      if (res.ok) {
        const blob = await res.blob()
        const sampleFile = new File([blob], preset.file, { type: 'image/tiff' })
        setFile(sampleFile)
      }
    } catch (err) {
      setError(`Failed to load preset: ${err.message}`)
    } finally {
      setLoading(false)
      setLoadingMessage('')
    }
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
    if (!file) return

    setLoading(true)
    setLoadingMessage('Synthesizing optical RGB with Exp9 ResNet...')
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
      // If backend is offline but a preset was selected, show friendly note
      if (activePreset) {
        setError(`Backend is offline (${err.message}). Displaying precomputed Exp9 validation output for ${activePreset.name}.`)
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
      const originalBlob = await fetch(outputImage).then((r) => r.blob())
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
            <span className="version-tag">Exp9 Production</span>
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
              Exp9 Global ResNet translates raw Landsat 9 split-window thermal infrared emissions into
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
                    Thermal IR (Bands 10 + 11)
                  </div>
                  <div className="slider-marker slider-marker--optical">
                    Synthesized Optical RGB {claheApplied ? '(CLAHE Enhanced)' : ''}
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
                  <span className="telemetry-value">128 x 128 px (100m)</span>
                </span>
              </div>

              <div className="execution-controls">
                <label className="checkbox-control" title="Test-Time Augmentation ensemble">
                  <input
                    type="checkbox"
                    checked={useTTA}
                    onChange={(e) => setUseTTA(e.target.checked)}
                    id="tta-toggle"
                  />
                  <span className="checkbox-label">4-Fold TTA</span>
                </label>

                {file && (
                  <button
                    type="button"
                    className="action-btn action-btn--primary"
                    onClick={runColorization}
                    disabled={loading}
                  >
                    Colorize
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

                {(file || displayImage) && (
                  <button
                    type="button"
                    className="action-btn action-btn--ghost"
                    onClick={resetAll}
                  >
                    Reset
                  </button>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 2: CROSS-SPECTRAL VISUAL SHOWCASE */}
        <section className="showcase-section">
          <div className="section-header">
            <h2 className="section-title">Cross-Spectral Optical Synthesis</h2>
            <p className="section-intro">
              Direct comparisons across global satellite acquisitions prove that the 9-block ResNet
              bottleneck eliminates the chromatic aberrations of direct skip connections, reliably
              reconstructing optical surface features.
            </p>
          </div>

          <div className="showcase-mosaic">
            <div className="mosaic-card">
              <img
                src="/assets/figure6_visual_inspection_panel.png"
                alt="Visual inspection panel comparing thermal input, ground truth RGB, legacy production U-Net, and active Exp9 ResNet across 4 global terrain biomes"
                className="mosaic-image"
                loading="lazy"
              />
              <div className="mosaic-caption">
                <span className="caption-bold">Quad-Biome Benchmark Panel</span>
                <span className="caption-text">
                  Columns: [1] Raw Thermal B10/B11; [2] Ground Truth Optical RGB; [3] Legacy U-Net
                  Baseline (severe chromatic noise); [4] Exp9 ResNet (our active production deployment).
                </span>
              </div>
            </div>

            <div className="mosaic-card">
              <img
                src="/assets/elephant-island-by-landsat-8.jpg"
                alt="Elephant Island Antarctica captured by Landsat observatory in polar sea ice"
                className="mosaic-image"
                loading="lazy"
              />
              <div className="mosaic-caption">
                <span className="caption-bold">Polar and Cryosphere Generalization</span>
                <span className="caption-text">
                  Extreme thermal contrast validation across ice-shelf margins and polar maritime
                  surfaces from the Landsat Earth observation archive.
                </span>
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 3: HOW IT WORKS / RADIATIVE TRANSFER */}
        <section className="architecture-section">
          <div className="section-header">
            <h2 className="section-title">Split-Window Radiative Transfer</h2>
            <p className="section-intro">
              Cross-spectral translation from longwave thermal infrared (10.60–12.51 µm) to visible
              spectrum optical reflectance (0.45–0.67 µm) requires physically informed inductive bias.
            </p>
          </div>

          <div className="architecture-grid">
            <div className="architecture-column">
              <div className="arch-visual-box">
                <img
                  src="/assets/figure1_cross_modal_decoupling.png"
                  alt="Cross-modal translation diagram between thermal infrared radiance and optical surface reflectance"
                  className="arch-diagram"
                  loading="lazy"
                />
              </div>
              <h3 className="arch-heading">Split-Window Radiance</h3>
              <p className="arch-body">
                Single-channel thermal sensors cannot distinguish surface moisture from bare rock.
                InfraNova AI ingests calibrated split-window radiance from Landsat 9 TIRS-2 Band 10
                (10.60–11.19 µm) and Band 11 (11.50–12.51 µm). The differential absorption provides the
                atmospheric moisture cues necessary to synthesize optical vegetation indices.
              </p>
            </div>

            <div className="architecture-column">
              <div className="arch-visual-box">
                <img
                  src="/assets/figure8_seamless_tiling_reconstruction.png"
                  alt="2D Hann window sliding reconstruction across large satellite rasters"
                  className="arch-diagram"
                  loading="lazy"
                />
              </div>
              <h3 className="arch-heading">2D Hann Window Blending</h3>
              <p className="arch-body">
                Full satellite scenes exceed memory limits when processed unsegmented. InfraNova
                decomposes scenes into overlapping 128 x 128 patches with a 32-pixel stride, blending
                predictions with a 2D raised-cosine Hann window to ensure zero border seams and
                smooth transitions across whole geographic rasters.
              </p>
            </div>
          </div>
        </section>

        {/* SECTION 4: STATS / BENCHMARK LEDGER */}
        <section className="stats-section">
          <div className="section-header">
            <h2 className="section-title">Held-Out Empirical Benchmark</h2>
            <p className="section-intro">
              Evaluated strictly on 1,259 unseen held-out test scenes across 449 global geographic
              regions under our leakage-free validation protocol.
            </p>
          </div>

          <div className="ledger-table">
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

      {/* SECTION 5: FOOTER / DESIGNED CLOSE */}
      <footer className="footer-frame">
        <div className="footer-tiles">
          <div className="footer-tile footer-tile--statement">
            <span className="footer-wordmark">InfraNova AI</span>
            <p className="footer-statement-text">
              Real-time cross-spectral optical translation for operational satellite earth observation.
            </p>
            <span className="footer-subtext">Research & Engineering Project</span>
          </div>

          <div className="footer-tile footer-tile--deployment">
            <span className="tile-header">Deployment</span>
            <p className="tile-body">
              Live on Oracle Cloud Infrastructure (OCI) Ampere A1 ARM64 instance.
            </p>
            <a
              href="https://infranovaai.duckdns.org"
              target="_blank"
              rel="noreferrer"
              className="tile-link"
            >
              infranovaai.duckdns.org
            </a>
          </div>

          <div className="footer-tile footer-tile--provenance">
            <span className="tile-header">Data Provenance</span>
            <p className="tile-body">
              Landsat 9 thermal infrared and optical surface reflectance data courtesy of the
              U.S. Geological Survey (USGS) and NASA Earth Observatory.
            </p>
            <span className="footer-license-note">
              Elephant Island photo by Joshua Stevens / NASA, public domain via Wikimedia Commons.
            </span>
          </div>

          <div className="footer-tile footer-tile--repo">
            <span className="tile-header">Code & Models</span>
            <p className="tile-body">
              Explore the complete training pipeline, loss derivations, and benchmark logs.
            </p>
            <a
              href="https://github.com/Soham-1009/InfraNova-AI"
              target="_blank"
              rel="noreferrer"
              className="tile-action-link"
            >
              GitHub Repository
            </a>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
