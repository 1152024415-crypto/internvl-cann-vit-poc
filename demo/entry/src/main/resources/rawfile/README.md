# HarmonyOS Rawfile Artifacts

Before building the HarmonyOS demo, this directory must contain:

- `internvl3_5_vit_projector_fp32_opset18_staticpos.om`
- `dog_pixel_values_fp32.bin`
- `dog_visual_tokens_fp32.bin`
- `dog.metadata.json`
- `cat_pixel_values_fp32.bin`
- `cat_visual_tokens_fp32.bin`
- `cat.metadata.json`
- `official_squeezenet_hiai.om`
- `official_labels_caffe.txt`
- `official_guitar_bgr_227.bin`
- `official_cup_bgr_227.bin`

The InternVL `.om` file is a large release artifact. Do not commit it to git;
fetch it from the release payload before building.

The `official_squeezenet_hiai.om` file comes from Huawei's public CANN Kit C++
sample. It is intentionally tracked in git because it is only about 2.5 MB and
provides a known-good CANN smoke test for the app, SDK, and device environment.
Expected size is 2,496,459 bytes and SHA256 is
`533B84458D7694174A220D3AA8B984B1F0021FB7D3997A1CC2732AA3B51C7AD3`.
The official `cup.jpg` and `guitar.jpg` media resources are also tracked under
`resources/base/media`. Their preprocessed 227 x 227 BGR planar inputs are
tracked here so the Native C++ validation can run the official classifier
without relying on ArkTS image preprocessing during environment diagnosis.

The four `.bin` files are tracked in git because they are small enough and make
yellow-zone validation reproducible after `git pull`.

The `.metadata.json` files are small and tracked here so the demo can fail
clearly if expected validation case metadata is missing.

If the validation tensors are regenerated, update the `.bin` and
`.metadata.json` files together.
