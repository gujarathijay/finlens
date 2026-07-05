import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";
import { type ExtractionResponse, type HistoryItem } from "./types";

const API = "http://localhost:8080";

const SAMPLE_TEXT = `The Company is subject to extensive regulatory requirements under the Dodd-Frank Act and Basel III framework. In September 2024, the SEC initiated an investigation into our derivatives trading practices, which remains ongoing as of the filing date. We have accrued $340 million in legal reserves related to this matter. Our total long-term debt stands at $12.8 billion, with $3.2 billion maturing in fiscal year 2025 and $4.1 billion maturing in 2026. The Company completed its acquisition of Pacific Financial Group on March 15, 2024, for approximately $2.7 billion in cash and stock, expanding our wealth management capabilities. Rising interest rates have compressed net interest margins by 45 basis points year-over-year, creating significant pressure on lending profitability.`;

function App() {
  const [text, setText] = useState(SAMPLE_TEXT);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ExtractionResponse | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await axios.get(`${API}/history?limit=5`);
      setHistory(res.data);
    } catch {
      // API not running yet — that's fine
    }
  };

  const handleExtract = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await axios.post(`${API}/extract`, { filing_text: text });
      setResult(res.data);
      fetchHistory();
    } catch (err: any) {
      setError(err.response?.data?.detail || "API error — is the server running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>
          <span>Fin</span>Lens
        </h1>
        <p>SEC Filing Structured Extraction</p>
      </header>

      {/* Input */}
      <div className="input-section">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste SEC filing text here..."
        />
        <button className="submit" onClick={handleExtract} disabled={loading}>
          {loading ? "Extracting..." : "Extract"}
        </button>
      </div>

      {/* Error */}
      {error && <div className="error">{error}</div>}

      {/* Results */}
      {result && result.extraction && (
        <div className="results">
          <h2>Extraction Results</h2>

          <div className="meta-bar">
            <span className={`badge ${result.status}`}>{result.status}</span>
            <span className="badge success">{result.latency_ms}ms</span>
            <span className="badge success">ID: {result.request_id}</span>
            {!result.guardrails_passed && (
              <span className="badge failed">
                Guardrails: {result.guardrail_failures.join(", ")}
              </span>
            )}
          </div>

          <div className="summary">{result.extraction.summary}</div>

          {/* Risk Factors */}
          {result.extraction.risk_factors.length > 0 && (
            <>
              <div className="section-title">
                Risk Factors ({result.extraction.risk_factors.length})
              </div>
              {result.extraction.risk_factors.map((r, i) => (
                <div className="card" key={i}>
                  <div className="value">{r.factor}</div>
                  <div style={{ marginTop: "0.5rem", display: "flex", gap: "1rem" }}>
                    <span className="label">
                      Category: <strong>{r.category}</strong>
                    </span>
                    <span className={`label severity-${r.severity}`}>
                      Severity: <strong>{r.severity}</strong>
                    </span>
                  </div>
                  <div className="label" style={{ marginTop: "0.3rem" }}>
                    Evidence: {r.evidence}
                  </div>
                </div>
              ))}
            </>
          )}

          {/* Material Events */}
          {result.extraction.material_events.length > 0 && (
            <>
              <div className="section-title">
                Material Events ({result.extraction.material_events.length})
              </div>
              {result.extraction.material_events.map((e, i) => (
                <div className="card" key={i}>
                  <div className="value">{e.event}</div>
                  <div style={{ marginTop: "0.5rem", display: "flex", gap: "1rem" }}>
                    {e.date && <span className="label">Date: {e.date}</span>}
                    <span className={`label impact-${e.impact}`}>
                      Impact: <strong>{e.impact}</strong>
                    </span>
                  </div>
                  <div className="label" style={{ marginTop: "0.3rem" }}>
                    {e.details}
                  </div>
                </div>
              ))}
            </>
          )}

          {/* Financial Obligations */}
          {result.extraction.financial_obligations.length > 0 && (
            <>
              <div className="section-title">
                Financial Obligations ({result.extraction.financial_obligations.length})
              </div>
              {result.extraction.financial_obligations.map((o, i) => (
                <div className="card" key={i}>
                  <div className="value">{o.obligation}</div>
                  <div style={{ marginTop: "0.5rem", display: "flex", gap: "1rem" }}>
                    {o.amount && <span className="label">Amount: <strong>{o.amount}</strong></span>}
                    {o.deadline && <span className="label">Deadline: {o.deadline}</span>}
                    <span className="label">Type: {o.category}</span>
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      )}

      {/* History */}
      {history.length > 0 && (
        <div className="history">
          <h2>Recent Extractions</h2>
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Company</th>
                <th>Status</th>
                <th>Risks</th>
                <th>Events</th>
                <th>Latency</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h) => (
                <tr key={h.id}>
                  <td>{h.id}</td>
                  <td>{h.company_name || "—"}</td>
                  <td>
                    <span className={`badge ${h.status}`}>{h.status}</span>
                  </td>
                  <td>{h.num_risks}</td>
                  <td>{h.num_events}</td>
                  <td>{h.latency_ms}ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default App;