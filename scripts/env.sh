#!/usr/bin/env bash
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-$(cat /root/.kaggle/access_token 2>/dev/null || true)}"
export PATH="/usr/local/bin:$PATH"
