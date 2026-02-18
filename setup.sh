#!/usr/bin/env bash
set -e

echo "=== LLM Country Bias Pilot — Setup ==="

# Install Python dependencies
pip install -r country_bias_pilot/requirements.txt

# HuggingFace auth (needed for gated models: Llama 3, Mistral)
if python -c "from huggingface_hub import HfFolder; assert HfFolder.get_token()" 2>/dev/null; then
    echo "HuggingFace token already configured."
else
    echo ""
    echo "Gated models (Llama 3, Mistral) require a HuggingFace token."
    echo "You can either:"
    echo "  1) Run: huggingface-cli login"
    echo "  2) Set: export HF_TOKEN=hf_..."
    echo ""
    huggingface-cli login
fi

echo ""
echo "Setup complete. Run the pipeline with:"
echo "  cd country_bias_pilot"
echo "  python run_pilot.py --test    # smoke test"
echo "  python run_pilot.py           # full run"
