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

export interface OfficialSmokeResult {
  ok: boolean;
  deviceType: string;
  errorStage: string;
  errorMessage: string;
  latencyMs: number;
  modelByteCount: number;
  inputCount: number;
  outputCount: number;
  inputByteCount: number;
  outputByteCount: number;
}

export interface OfficialClassificationResult {
  ok: boolean;
  imageName: string;
  deviceType: string;
  errorStage: string;
  errorMessage: string;
  latencyMs: number;
  inputByteCount: number;
  outputElementCount: number;
  top1Label: string;
  top1Score: number;
  top2Label: string;
  top2Score: number;
  top3Label: string;
  top3Score: number;
}

export const listTestCases: () => string[];
export const loadModel: (resourceManager: object) => Promise<ModelStatusResult>;
export const unloadModel: () => Promise<ModelStatusResult>;
export const runOfficialSmoke: (resourceManager: object) => Promise<OfficialSmokeResult>;
export const runOfficialClassification: (resourceManager: object, imageName: string) => Promise<OfficialClassificationResult>;
export const runOnce: (resourceManager: object, caseName: string) => Promise<RunResult>;
export const runStability: (resourceManager: object, caseName: string, repeatCount: number) => Promise<StabilityResult>;
