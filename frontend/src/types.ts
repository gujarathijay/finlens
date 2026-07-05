export interface RiskFactor {
  factor: string;
  category: string;
  severity: "high" | "medium" | "low";
  evidence: string;
}

export interface MaterialEvent {
  event: string;
  date: string | null;
  impact: "positive" | "negative" | "neutral";
  details: string;
}

export interface FinancialObligation {
  obligation: string;
  amount: string | null;
  deadline: string | null;
  category: string;
}

export interface ExtractionResult {
  company_name: string;
  filing_type: string;
  fiscal_year: string;
  risk_factors: RiskFactor[];
  material_events: MaterialEvent[];
  financial_obligations: FinancialObligation[];
  summary: string;
}

export interface ExtractionResponse {
  status: "success" | "failed" | "flagged";
  extraction: ExtractionResult | null;
  guardrails_passed: boolean;
  guardrail_failures: string[];
  latency_ms: number;
  request_id: number;
}

export interface HistoryItem {
  id: number;
  created_at: string;
  company_name: string | null;
  status: string;
  num_risks: number;
  num_events: number;
  num_obligations: number;
  guardrails_passed: boolean;
  latency_ms: number;
}