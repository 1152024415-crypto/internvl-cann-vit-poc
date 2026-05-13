export interface RunResult {
  ok: boolean;
  caseName: string;
  deviceType: string;
  errorStage: string;
  errorMessage: string;
  latencyMs: number;
  outputElementCount: number;
  maxAbsDiff: number;
  meanAbsDiff: number;
  cosine: number;
  finite: boolean;
}

export interface StabilityResult {
  ok: boolean;
  caseName: string;
  repeatCount: number;
  successCount: number;
  minLatencyMs: number;
  maxLatencyMs: number;
  avgLatencyMs: number;
  errorStage: string;
  errorMessage: string;
  lastRun: RunResult;
}

export const listTestCases: () => string[];
export const runOnce: (resourceManager: object, caseName: string) => RunResult;
export const runStability: (resourceManager: object, caseName: string, repeatCount: number) => StabilityResult;
