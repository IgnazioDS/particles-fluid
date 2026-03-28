# ─────────────────────────────────────────────────────────────
# Particles Fluid  —  Makefile
# ─────────────────────────────────────────────────────────────

.PHONY: install run run-gpu run-demo run-fast clean help

# ── Setup ─────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Run targets ───────────────────────────────────────────────
run:                        ## Default: webcam + CPU Taichi
	python main.py

run-gpu:                    ## GPU backend (CUDA / Metal)
	python main.py --gpu

run-demo:                   ## Mouse-only demo (no webcam)
	python main.py --no-webcam

run-fast:                   ## Fewer particles for slower machines
	python main.py --particles 800

run-hd:                     ## More particles for powerful machines
	python main.py --particles 2500

# ── Maintenance ───────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

help:
	@echo ""
	@echo "  make install     — install Python dependencies"
	@echo "  make run         — launch with webcam (CPU)"
	@echo "  make run-gpu     — launch with webcam (GPU/CUDA)"
	@echo "  make run-demo    — launch mouse-only demo"
	@echo "  make run-fast    — 800 particles (slower machines)"
	@echo "  make run-hd      — 2500 particles (fast machines)"
	@echo ""
