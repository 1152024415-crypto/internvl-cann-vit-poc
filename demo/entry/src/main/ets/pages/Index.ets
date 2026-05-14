import { hilog } from '@kit.PerformanceAnalysisKit';
import testNapi, { ModelStatusResult, OfficialSmokeResult, RunResult, StabilityResult } from 'libentry.so';

const DOMAIN = 0x0000;
const TAG = 'InternVLValidation';

@Entry
@Component
struct Index {
  @State selectedCase: string = '';
  @State cases: string[] = [];
  @State busy: boolean = false;
  @State modelLoaded: boolean = false;
  @State status: string = 'Ready';
  @State modelText: string = '';
  @State runText: string = '';
  @State stabilityText: string = '';

  aboutToAppear(): void {
    try {
      const nativeCases = testNapi.listTestCases();
      if (nativeCases.length > 0) {
        this.cases = nativeCases;
        this.selectedCase = nativeCases[0];
      }
    } catch (error) {
      const message = `${error}`;
      this.status = 'Failed to list native test cases';
      this.runText = message;
      hilog.error(DOMAIN, TAG, 'listTestCases error: %{public}s', message);
    }
  }

  formatModel(result: ModelStatusResult): string {
    return [
      `ok: ${result.ok}`,
      `loaded: ${result.loaded}`,
      `device: ${result.deviceType}`,
      `latency: ${result.latencyMs} ms`,
      `error_stage: ${result.errorStage}`,
      `error_message: ${result.errorMessage}`
    ].join('\n');
  }

  formatOfficialSmoke(result: OfficialSmokeResult): string {
    return [
      `ok: ${result.ok}`,
      `device: ${result.deviceType}`,
      `latency: ${result.latencyMs} ms`,
      `model_bytes: ${result.modelByteCount}`,
      `input_count: ${result.inputCount}`,
      `output_count: ${result.outputCount}`,
      `input_bytes: ${result.inputByteCount}`,
      `output_bytes: ${result.outputByteCount}`,
      `error_stage: ${result.errorStage}`,
      `error_message: ${result.errorMessage}`
    ].join('\n');
  }

  formatRun(result: RunResult): string {
    return [
      `ok: ${result.ok}`,
      `case: ${result.caseName}`,
      `device: ${result.deviceType}`,
      `latency: ${result.latencyMs} ms`,
      `output_elements: ${result.outputElementCount}`,
      `output_shape: ${result.outputShape}`,
      `output_shape_ok: ${result.outputShapeOk}`,
      `finite: ${result.finite}`,
      `cosine: ${result.cosine}`,
      `mean_abs_diff: ${result.meanAbsDiff}`,
      `max_abs_diff: ${result.maxAbsDiff}`,
      `error_stage: ${result.errorStage}`,
      `error_message: ${result.errorMessage}`
    ].join('\n');
  }

  formatStability(result: StabilityResult): string {
    return [
      `ok: ${result.ok}`,
      `case: ${result.caseName}`,
      `repeat: ${result.repeatCount}`,
      `success: ${result.successCount}`,
      `avg: ${result.avgLatencyMs} ms`,
      `min: ${result.minLatencyMs} ms`,
      `max: ${result.maxLatencyMs} ms`,
      `error_stage: ${result.errorStage}`,
      `error_message: ${result.errorMessage}`,
      '',
      'last_run:',
      this.formatRun(result.lastRun)
    ].join('\n');
  }

  async loadModel(): Promise<void> {
    if (this.busy) {
      return;
    }

    this.busy = true;
    this.status = 'Loading OM model...';
    try {
      const result = await testNapi.loadModel(getContext(this).resourceManager);
      this.modelLoaded = result.loaded;
      this.modelText = this.formatModel(result);
      this.status = result.ok ? 'Model loaded' : 'Model load failed';
      hilog.info(DOMAIN, TAG, 'loadModel ok=%{public}s loaded=%{public}s', `${result.ok}`, `${result.loaded}`);
    } catch (error) {
      const message = `${error}`;
      this.modelLoaded = false;
      this.status = 'Model load error';
      this.modelText = message;
      hilog.error(DOMAIN, TAG, 'loadModel error: %{public}s', message);
    } finally {
      this.busy = false;
    }
  }

  async unloadModel(): Promise<void> {
    if (this.busy) {
      return;
    }

    this.busy = true;
    this.status = 'Unloading model...';
    try {
      const result = await testNapi.unloadModel();
      this.modelLoaded = result.loaded;
      this.modelText = this.formatModel(result);
      this.status = result.ok ? 'Model unloaded' : 'Model unload failed';
      hilog.info(DOMAIN, TAG, 'unloadModel ok=%{public}s loaded=%{public}s', `${result.ok}`, `${result.loaded}`);
    } catch (error) {
      const message = `${error}`;
      this.status = 'Model unload error';
      this.modelText = message;
      hilog.error(DOMAIN, TAG, 'unloadModel error: %{public}s', message);
    } finally {
      this.busy = false;
    }
  }

  async runOfficialSmoke(): Promise<void> {
    if (this.busy) {
      return;
    }

    this.busy = true;
    this.status = 'Running official OM smoke test...';
    try {
      const result = await testNapi.runOfficialSmoke(getContext(this).resourceManager);
      this.modelText = this.formatOfficialSmoke(result);
      this.status = result.ok ? 'Official smoke passed' : 'Official smoke failed';
      hilog.info(DOMAIN, TAG, 'runOfficialSmoke ok=%{public}s', `${result.ok}`);
    } catch (error) {
      const message = `${error}`;
      this.status = 'Official smoke error';
      this.modelText = message;
      hilog.error(DOMAIN, TAG, 'runOfficialSmoke error: %{public}s', message);
    } finally {
      this.busy = false;
    }
  }

  async runOnce(): Promise<void> {
    if (this.busy) {
      return;
    }
    if (this.selectedCase.length === 0) {
      this.status = 'No native test case available';
      return;
    }
    if (!this.modelLoaded) {
      this.status = 'Load model first';
      return;
    }

    this.busy = true;
    this.status = `Running single ${this.selectedCase} case...`;
    try {
      const result = await testNapi.runOnce(getContext(this).resourceManager, this.selectedCase);
      this.runText = this.formatRun(result);
      this.status = result.ok ? 'Single run passed' : 'Single run failed';
      hilog.info(DOMAIN, TAG, 'runOnce %{public}s ok=%{public}s', this.selectedCase, `${result.ok}`);
    } catch (error) {
      const message = `${error}`;
      this.status = 'Single run error';
      this.runText = message;
      hilog.error(DOMAIN, TAG, 'runOnce error: %{public}s', message);
    } finally {
      this.busy = false;
    }
  }

  async runStability(): Promise<void> {
    if (this.busy) {
      return;
    }
    if (this.selectedCase.length === 0) {
      this.status = 'No native test case available';
      return;
    }
    if (!this.modelLoaded) {
      this.status = 'Load model first';
      return;
    }

    this.busy = true;
    this.status = `Running 20x ${this.selectedCase} stability...`;
    try {
      const result = await testNapi.runStability(getContext(this).resourceManager, this.selectedCase, 20);
      this.stabilityText = this.formatStability(result);
      this.status = result.ok ? 'Stability run passed' : 'Stability run failed';
      hilog.info(DOMAIN, TAG, 'runStability %{public}s ok=%{public}s', this.selectedCase, `${result.ok}`);
    } catch (error) {
      const message = `${error}`;
      this.status = 'Stability run error';
      this.stabilityText = message;
      hilog.error(DOMAIN, TAG, 'runStability error: %{public}s', message);
    } finally {
      this.busy = false;
    }
  }

  build() {
    Column({ space: 16 }) {
      Text('InternVL OM Validation')
        .fontSize(24)
        .fontWeight(FontWeight.Bold)
        .width('100%')

      Row({ space: 12 }) {
        ForEach(this.cases, (caseName: string) => {
          Button(caseName)
            .enabled(!this.busy)
            .onClick(() => {
              this.selectedCase = caseName;
              this.status = 'Ready';
            })
        }, (caseName: string) => caseName)
      }
      .width('100%')

      Text(`Selected case: ${this.selectedCase}`)
        .fontSize(16)
        .width('100%')

      Row({ space: 12 }) {
        Button('Load Model')
          .enabled(!this.busy)
          .onClick(() => {
            this.loadModel();
          })

        Button('Unload')
          .enabled(!this.busy && this.modelLoaded)
          .onClick(() => {
            this.unloadModel();
          })

        Button('Official Smoke')
          .enabled(!this.busy)
          .onClick(() => {
            this.runOfficialSmoke();
          })
      }
      .width('100%')

      Row({ space: 12 }) {
        Button('Run Once')
          .enabled(!this.busy && this.modelLoaded)
          .onClick(() => {
            this.runOnce();
          })

        Button('Run 20 Stability')
          .enabled(!this.busy && this.modelLoaded)
          .onClick(() => {
            this.runStability();
          })
      }
      .width('100%')

      Text(this.status)
        .fontSize(16)
        .fontWeight(FontWeight.Medium)
        .width('100%')

      Text('Model')
        .fontSize(18)
        .fontWeight(FontWeight.Medium)
        .width('100%')

      TextArea({ text: this.modelText })
        .height(120)
        .width('100%')

      Text('Single Run')
        .fontSize(18)
        .fontWeight(FontWeight.Medium)
        .width('100%')

      TextArea({ text: this.runText })
        .height(180)
        .width('100%')

      Text('Stability')
        .fontSize(18)
        .fontWeight(FontWeight.Medium)
        .width('100%')

      TextArea({ text: this.stabilityText })
        .height(220)
        .width('100%')
    }
    .padding(20)
    .width('100%')
    .height('100%')
  }
}
