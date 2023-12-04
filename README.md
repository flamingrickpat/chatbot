# chatbot

To use on windows you must install pytorch, flash attention 2 and exllamav2. In my case that was:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# https://github.com/jllllll/flash-attention/releases
pip3 install flash_attn-2.3.6+cu121torch2.1cxx11abiFALSE-cp310-cp310-win_amd64.whl
# https://github.com/turboderp/exllamav2/releases
pip3 install exllamav2-0.0.10+cu121-cp310-cp310-win_amd64.whl
```