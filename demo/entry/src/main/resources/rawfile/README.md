# HarmonyOS Rawfile Artifacts

Before building the HarmonyOS demo, place the yellow-zone release artifacts in
this directory:

- `internvl3_5_vit_projector_fp32_opset18_staticpos.om`
- `dog_pixel_values_fp32.bin`
- `dog_visual_tokens_fp32.bin`
- `dog.metadata.json`
- `cat_pixel_values_fp32.bin`
- `cat_visual_tokens_fp32.bin`
- `cat.metadata.json`

The `.om` and `.bin` files are large release artifacts. Do not commit them to
git; fetch them from the release payload before building.
