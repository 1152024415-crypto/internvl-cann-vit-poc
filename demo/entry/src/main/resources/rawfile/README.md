# HarmonyOS Rawfile Artifacts

Before building the HarmonyOS demo, this directory must contain:

- `internvl3_5_vit_projector_fp32_opset18_staticpos.om`
- `dog_pixel_values_fp32.bin`
- `dog_visual_tokens_fp32.bin`
- `dog.metadata.json`
- `cat_pixel_values_fp32.bin`
- `cat_visual_tokens_fp32.bin`
- `cat.metadata.json`

The `.om` file is a large release artifact. Do not commit it to git; fetch it
from the release payload before building.

The four `.bin` files are tracked in git because they are small enough and make
yellow-zone validation reproducible after `git pull`.

The `.metadata.json` files are small and tracked here so the demo can fail
clearly if expected validation case metadata is missing.

If the validation tensors are regenerated, update the `.bin` and
`.metadata.json` files together.
