---
name: ShellShot model loading
description: Compatibility rules for adding saved Keras checkpoints to ShellShot.
---

Saved Keras checkpoints are not guaranteed to share one input resolution or deserialize without custom objects. Register the model's serialized preprocessing function and derive image sizing from the loaded model rather than assuming 224×224.

**Why:** A supplied EfficientNetV2S checkpoint used a named `preprocess_input` Lambda and required 384×384 input; without both compatibility steps it appeared unavailable or failed during prediction.

**How to apply:** When adding a checkpoint, verify its input shape, output class count/order, and any serialized custom functions before declaring it ready.