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
  outputShapeOk: boolean;
  outputShape: string;
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

export interface ModelStatusResult {
  ok: boolean;
  loaded: boolean;
  deviceType: string;
  errorStage: string;
  errorMessage: string;
  latencyMs: number;
}

export const listTestCases: () => string[];
export const loadModel: (resourceManager: object) => Promise<ModelStatusResult>;
export const unloadModel: () => Promise<ModelStatusResult>;
export const runOnce: (resourceManager: object, caseName: string) => Promise<RunResult>;
export const runStability: (resourceManager: object, caseName: string, repeatCount: number) => Promise<StabilityResult>;
