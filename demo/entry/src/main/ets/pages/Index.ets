import { hilog } from '@kit.PerformanceAnalysisKit';
import testNapi, { RunResult, StabilityResult } from 'libentry.so';

const DOMAIN = 0x0000;
const TAG = 'InternVLValidation';

@Entry
@Component
struct Index {
  @State selectedCase: string = 'dog';
  @State cases: string[] = ['dog', 'cat'];
  @State status: string = 'Ready';
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

  formatRun(result: RunResult): string {
    return [
      `ok: ${result.ok}`,
      `case: ${result.caseName}`,
      `device: ${result.deviceType}`,
      `latency: ${result.latencyMs} ms`,
      `output_elements: ${result.outputElementCount}`,
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

  runOnce(): void {
    this.status = `Running single ${this.selectedCase} case...`;
    try {
      const result = testNapi.runOnce(getContext(this).resourceManager, this.selectedCase);
      this.runText = this.formatRun(result);
      this.status = result.ok ? 'Single run passed' : 'Single run failed';
      hilog.info(DOMAIN, TAG, 'runOnce %{public}s ok=%{public}s', this.selectedCase, `${result.ok}`);
    } catch (error) {
      const message = `${error}`;
      this.status = 'Single run error';
      this.runText = message;
      hilog.error(DOMAIN, TAG, 'runOnce error: %{public}s', message);
    }
  }

  runStability(): void {
    this.status = `Running 20x ${this.selectedCase} stability...`;
    try {
      const result = testNapi.runStability(getContext(this).resourceManager, this.selectedCase, 20);
      this.stabilityText = this.formatStability(result);
      this.status = result.ok ? 'Stability run passed' : 'Stability run failed';
      hilog.info(DOMAIN, TAG, 'runStability %{public}s ok=%{public}s', this.selectedCase, `${result.ok}`);
    } catch (error) {
      const message = `${error}`;
      this.status = 'Stability run error';
      this.stabilityText = message;
      hilog.error(DOMAIN, TAG, 'runStability error: %{public}s', message);
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
        Button('Run Once')
          .onClick(() => {
            this.runOnce();
          })

        Button('Run 20 Stability')
          .onClick(() => {
            this.runStability();
          })
      }
      .width('100%')

      Text(this.status)
        .fontSize(16)
        .fontWeight(FontWeight.Medium)
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
