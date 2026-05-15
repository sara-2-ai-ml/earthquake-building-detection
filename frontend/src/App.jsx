import { useState, useRef } from "react"
import axios from "axios"
import "./App.css"

const API_URL = "http://127.0.0.1:8000"

export default function App() {
  const [image, setImage]       = useState(null)
  const [preview, setPreview]   = useState(null)
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [dragging, setDragging] = useState(false)
  const inputRef                = useRef(null)

  const handleFile = (file) => {
    if (!file) return
    setImage(file)
    setPreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
  }

  const handleUpload   = (e) => handleFile(e.target.files[0])
  const handleDrop     = (e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]) }
  const handleDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const handleDragLeave= ()  => setDragging(false)

  const handlePredict = async () => {
    if (!image) return
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append("file", image)
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      setResult(response.data)
    } catch {
      setError("Prediction failed — make sure the backend is running.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">

      {/* ── Hero Section ── */}
      <section className="hero">
        <div className="hero-bg">
          <img src="/satellite.png" alt="satellite" className="hero-satellite"/>
          <div className="hero-overlay"/>
        </div>
        <div className="hero-content">
          <div className="hero-tag">YOLOv11 · Instance Segmentation</div>
          <h1 className="hero-title">
            Earthquake Building<br/>Damage Detection
          </h1>
          <p className="hero-sub">
            Automated satellite imagery analysis for rapid disaster response.<br/>
            Türkiye 2023 · Hatay · Adıyaman · Kahramanmaraş
          </p>
          <div className="hero-stats">
            <div className="hero-stat">
              <span className="hero-stat-num">0.808</span>
              <span className="hero-stat-label">Mask mAP50</span>
            </div>
            <div className="hero-divider"/>
            <div className="hero-stat">
              <span className="hero-stat-num">0.753</span>
              <span className="hero-stat-label">Recall</span>
            </div>
            <div className="hero-divider"/>
            <div className="hero-stat">
              <span className="hero-stat-num">80</span>
              <span className="hero-stat-label">Training Images</span>
            </div>
            <div className="hero-divider"/>
            <div className="hero-stat">
              <span className="hero-stat-num">2</span>
              <span className="hero-stat-label">Classes</span>
            </div>
          </div>
        </div>
        <div className="hero-scroll">↓ Upload a satellite image below</div>
      </section>

      {/* ── Main Section ── */}
      <section className="main-section">
        <div className="main-inner">

          {/* Upload Panel */}
          <div className="panel">
            <div className="panel-label">Satellite Image</div>

            <div
              className={`upload-zone ${dragging ? "dragging" : ""} ${preview ? "has-image" : ""}`}
              onClick={() => inputRef.current.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
            >
              {preview
                ? <img src={preview} alt="preview" className="preview-img"/>
                : (
                  <div className="upload-placeholder">
                    <div className="upload-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
                      </svg>
                    </div>
                    <p className="upload-text">Drop satellite image here</p>
                    <p className="upload-sub">or click to browse · JPG, PNG</p>
                  </div>
                )
              }
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                onChange={handleUpload}
                style={{ display: "none" }}
              />
            </div>

            <button
              className={`predict-btn ${loading ? "loading" : ""}`}
              onClick={handlePredict}
              disabled={!image || loading}
            >
              {loading
                ? <><span className="spinner"/><span>Analyzing imagery...</span></>
                : "Run Detection"
              }
            </button>

            {error && <p className="error">{error}</p>}

            <div className="model-info">
              <div className="info-row">
                <span className="info-label">Model</span>
                <span className="info-value">YOLOv11m-seg</span>
              </div>
              <div className="info-row">
                <span className="info-label">Optimizer</span>
                <span className="info-value">AdamW · 150 epochs</span>
              </div>
              <div className="info-row">
                <span className="info-label">Mask mAP50</span>
                <span className="info-value accent">0.808</span>
              </div>
              <div className="info-row">
                <span className="info-label">Recall</span>
                <span className="info-value accent">0.753</span>
              </div>
              <div className="info-row">
                <span className="info-label">Confidence</span>
                <span className="info-value">0.25 threshold</span>
              </div>
            </div>
          </div>

          {/* Results Panel */}
          <div className="panel">
            <div className="panel-label">Detection Results</div>

            {!result && !loading && (
              <div className="empty-state">
                <p>Upload a satellite image and run detection to see results here.</p>
              </div>
            )}

            {loading && (
              <div className="analyzing">
                <div className="pulse-ring"/>
                <p>Analyzing satellite imagery...</p>
              </div>
            )}

            {result && (
              <>
                <div className="result-img-wrap">
                  <img
                    src={`data:image/jpeg;base64,${result.annotated_image}`}
                    alt="annotated"
                    className="result-img"
                  />
                </div>

                <div className="stats">
                  <div className="stat-card collapsed">
                    <span className="stat-number">{result.counts.collapsed}</span>
                    <span className="stat-label">Collapsed</span>
                  </div>
                  <div className="stat-card intact">
                    <span className="stat-number">{result.counts.intact}</span>
                    <span className="stat-label">Intact</span>
                  </div>
                  <div className="stat-card total">
                    <span className="stat-number">{result.total}</span>
                    <span className="stat-label">Total</span>
                  </div>
                </div>

                <div className="damage-bar-wrap">
                  <div className="damage-bar-label">
                    <span>Damage ratio</span>
                    <span>{Math.round((result.counts.collapsed / result.total) * 100)}%</span>
                  </div>
                  <div className="damage-bar">
                    <div
                      className="damage-fill"
                      style={{ width: `${(result.counts.collapsed / result.total) * 100}%` }}
                    />
                  </div>
                </div>

                <div className="detection-list">
                  <div className="panel-label small">Detections</div>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Class</th>
                          <th>Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.detections.map((det, i) => (
                          <tr key={i}>
                            <td>{i + 1}</td>
                            <td><span className={`badge ${det.class}`}>{det.class}</span></td>
                            <td>
                              <div className="conf-wrap">
                                <div className="conf-bar">
                                  <div className={`conf-fill ${det.class}`}
                                    style={{ width: `${det.confidence * 100}%` }}/>
                                </div>
                                <span>{(det.confidence * 100).toFixed(1)}%</span>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="footer">
        <p>Earthquake Building Damage Detection · YOLOv11 · Türkiye 2023</p>
        <p>Hatay · Adıyaman · Kahramanmaraş · Maxar Satellite Imagery</p>
      </footer>
    </div>
  )
}